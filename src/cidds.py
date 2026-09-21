"""CIDDS-001 support: protocol normalization, the bucketed alphabet, and semantic checks.

Why this module exists
----------------------
The framework was built for generators (GPT-2, WGAN) that emit NetFlow records character by
character, so a generated address is a string that may or may not be a valid IPv4 address and
a generated port may be any integer. Two design decisions follow from that and neither holds
for the CIDDS/TabDDPM pipeline in ../mp-lissmann:

  1. `syntax_check.check_netflows` treats a non-parseable address or port as a generation
     error. TabDDPM emits a one-hot over 37 IP slots and 19 port slots, so its address is
     valid by construction -- and 89.8 % of CIDDS destination addresses were bucketed to the
     single token 'External' before training, together with 93.5 % of source ports to
     '>1024'. Judged by the stock check, 100 % of the flows are errors, and the number
     measures ../mp-lissmann/prework/prep_modified.py rather than the generative model.

  2. `binarize_queensland.split_df_seperator` splits addresses per octet and binarizes ports
     to 16 bits. 'External' has no octets and '>1024' is not a number, so both silently
     become 0.0.0.0 / port 0 (`decimalToBinary` swallows the exception and returns zeros).
     The ML metrics then run to completion on a feature space where every external host and
     every ephemeral port is the same point.

Both are handled here instead of by widening the generic code paths, so the Queensland
datasets keep behaving exactly as before.

The alphabet is never defined in this file. It is read from the `ip_vocab.json` /
`port_vocab.json` that prep_modified.py wrote and that sample.py inverts when decoding
generated ids. If this module invented its own token list it could drift from the data
without anything raising -- the failure mode would be a plausible-looking wrong number.
"""

import json
import numpy as np
import pandas as pd

import src.binarize_queensland as bq


# The CIDDS export carries protocol names; NetFlow (and encode_protocol_one_hot, and
# check_number_float) want the IP protocol byte. Applied to real and synthetic alike -- if
# only one side were converted the two would disagree on every row of a categorical
# attribute whose distributions actually match.
PROTOCOL_NUMBERS = {'TCP': 6, 'UDP': 17, 'ICMP': 1, 'IGMP': 2}

UDP = 17

# Smallest Ethernet frame carrying an IP packet. A flow reporting more packets than
# bytes/MIN_FRAME_BYTES is not a flow that could have been captured.
MIN_FRAME_BYTES = 28

# The bidirectional CIDDS flows added Source/Destination connection-count features that the
# 12-column Queensland NetFlow schema has no slot for, so RQ2 never measured them (mp-lissmann
# docs/remaining_tasks_plan.md Task 3). The mp-lissmann exporter/writer now append them (after
# 'Label'); when INCLUDE_EXTRA_NUMERIC is switched on -- via
# data_metrics_framework.enable_cidds_extra_numeric(), which the *_cmd.py scripts call only on
# --extra_numeric -- they are folded into the discriminator feature space here and into the
# marginal/dependency attribute lists there. Default OFF, so every canonical CIDDS and Queensland
# run is byte-identical to before (the same discipline as --vocab_dir itself).
CIDDS_EXTRA_NUMERIC = ['Src Conns', 'Dst Conns']
INCLUDE_EXTRA_NUMERIC = False


def normalize_protocol(df):
    """Map protocol names onto protocol bytes, leaving already-numeric columns alone."""
    if 'PROTOCOL' not in df.columns:
        return df
    if not df['PROTOCOL'].dtype == object:
        return df
    mapped = df['PROTOCOL'].map(PROTOCOL_NUMBERS)
    # Anything unmapped was already numeric-as-string, or is a generation error; coerce and
    # leave the error for the syntax check to count rather than silently rewriting it.
    df = df.copy()
    df['PROTOCOL'] = mapped.fillna(pd.to_numeric(df['PROTOCOL'], errors='coerce'))
    return df


def load_alphabets(vocab_dir):
    """Read the IP and port token lists from the vocabularies the training data was built with.

    Returned in slot order so the one-hot column layout is deterministic across datasets --
    the discriminator trains on one frame and predicts on the other, so a column ordering
    that depended on which values happened to occur would compare different features.
    """
    ip_vocab = json.loads((vocab_dir / 'ip_vocab.json').read_text())
    port_vocab = json.loads((vocab_dir / 'port_vocab.json').read_text())
    ip_tokens = [k for k, _ in sorted(ip_vocab.items(), key=lambda kv: kv[1])]
    port_tokens = [k for k, _ in sorted(port_vocab.items(), key=lambda kv: kv[1])]
    return ip_tokens, port_tokens


def encode_dataset_cidds(df_orig, ip_tokens, port_tokens):
    """Binarize a CIDDS frame, one-hotting the IP/port tokens instead of octets and bits.

    Everything other than the four address/port columns is encoded exactly as
    `data_metrics_framework.encode_dataset` does, so the two feature spaces stay comparable
    in every dimension that the bucketing does not touch.
    """
    df = df_orig.copy()
    if ('L7_PROTO' in df.columns) and ('Attack' in df.columns):
        df = df.drop(columns=['L7_PROTO', 'Attack'])
    df = normalize_protocol(df)

    parts = []
    for col, tokens in [('IPV4_SRC_ADDR', ip_tokens), ('L4_SRC_PORT', port_tokens),
                        ('IPV4_DST_ADDR', ip_tokens), ('L4_DST_PORT', port_tokens)]:
        text = df[col].astype(str)
        parts.append(pd.DataFrame({f'{col}_{t}': (text == t).astype('int8') for t in tokens},
                                  index=df.index))

    parts.append(bq.encode_protocol_one_hot(df, 'PROTOCOL'))

    byte_bins = bq.create_bins_log_bits(4 * 8)
    for col in ['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS']:
        parts.append(bq.quantisize_values_to_bins(df.copy(), col, bin_list=byte_bins))

    parts.append(bq.binarize_and_split(df.copy(), 'TCP_FLAGS', bits=8))
    parts.append(bq.quantisize_values_to_bins(
        df.copy(), 'FLOW_DURATION_MILLISECONDS',
        bin_list=bq.create_bins_linear_max(4.294967e+06, 16)))

    # Connection counts are integer counts with the same heavy-tailed shape as the byte/packet
    # columns, so they reuse the log-spaced byte bins. Gated so canonical runs are unchanged.
    if INCLUDE_EXTRA_NUMERIC:
        for col in CIDDS_EXTRA_NUMERIC:
            parts.append(bq.quantisize_values_to_bins(df.copy(), col, bin_list=byte_bins))

    parts.append(df['Label'])

    encoded = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    return encoded.astype('int8')


# --- semantic validity -------------------------------------------------------------------

def _as_float(series):
    return pd.to_numeric(series, errors='coerce')


def check_netflows_cidds(df, ip_tokens, port_tokens):
    """Count flows that are not physically realisable NetFlow records.

    The format checks the stock `check_netflows` performs are dropped: an address or port
    outside the vocabulary is impossible for this generator, so counting it would report a
    constant. What is kept, and extended, are the cross-field invariants -- constraints the
    model genuinely can violate and does. Measured on the real train split these hold with
    zero exceptions in 300,000 rows, so every violation on the synthetic side is a real
    generation error rather than a rare-but-legal record:

        packets and bytes agree per direction (one is zero iff the other is)
        bytes >= packets * MIN_FRAME_BYTES per direction
        the flow carries at least one packet and one byte overall
        a UDP flow carries no TCP flags
        counts are non-negative, Label is 0 or 1
        addresses and ports are inside the vocabulary alphabet

    The last one is a guard rather than a quality signal: a token outside the alphabet means
    the file was produced against a different vocabulary than the one loaded here, which
    would otherwise surface as a quietly inflated distance.
    """
    counts = {'Flows': 0}
    for key in ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'PROTOCOL',
                'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS',
                'FLOW_DURATION_MILLISECONDS', 'Label', 'In_out_bytes', 'In_out_pckts',
                'Protocol_flags', 'Pkts_bytes_agree', 'Min_frame_size']:
        counts[key] = 0

    df = normalize_protocol(df)
    ip_set, port_set = set(ip_tokens), set(port_tokens)

    in_bytes, out_bytes = _as_float(df['IN_BYTES']), _as_float(df['OUT_BYTES'])
    in_pkts, out_pkts = _as_float(df['IN_PKTS']), _as_float(df['OUT_PKTS'])
    flags = _as_float(df['TCP_FLAGS'])
    duration = _as_float(df['FLOW_DURATION_MILLISECONDS'])
    proto = _as_float(df['PROTOCOL'])
    label = _as_float(df['Label'])

    bad = {}
    bad['IPV4_SRC_ADDR'] = ~df['IPV4_SRC_ADDR'].astype(str).isin(ip_set)
    bad['IPV4_DST_ADDR'] = ~df['IPV4_DST_ADDR'].astype(str).isin(ip_set)
    bad['L4_SRC_PORT'] = ~df['L4_SRC_PORT'].astype(str).isin(port_set)
    bad['L4_DST_PORT'] = ~df['L4_DST_PORT'].astype(str).isin(port_set)
    bad['PROTOCOL'] = proto.isna() | (proto < 0)
    for name, values in [('IN_BYTES', in_bytes), ('OUT_BYTES', out_bytes),
                         ('IN_PKTS', in_pkts), ('OUT_PKTS', out_pkts),
                         ('TCP_FLAGS', flags), ('FLOW_DURATION_MILLISECONDS', duration)]:
        bad[name] = values.isna() | (values < 0)
    bad['Label'] = ~label.isin([0, 1])

    bad['In_out_bytes'] = (in_bytes.fillna(0) + out_bytes.fillna(0)) <= 0
    bad['In_out_pckts'] = (in_pkts.fillna(0) + out_pkts.fillna(0)) <= 0
    bad['Protocol_flags'] = (proto == UDP) & (flags > 0)

    # One direction reporting packets but no bytes (or the reverse) cannot happen on the wire.
    bad['Pkts_bytes_agree'] = (((in_pkts == 0) != (in_bytes == 0)) |
                               ((out_pkts == 0) != (out_bytes == 0))).fillna(False)
    bad['Min_frame_size'] = ((in_bytes < in_pkts * MIN_FRAME_BYTES) |
                             (out_bytes < out_pkts * MIN_FRAME_BYTES)).fillna(False)

    invalid = np.zeros(len(df), dtype=bool)
    for key, mask in bad.items():
        mask = mask.fillna(True).to_numpy()
        counts[key] = int(mask.sum())
        invalid |= mask
    counts['Flows'] = int(invalid.sum())

    return df.loc[~invalid].reset_index(drop=True), counts
