from remap_distillation_targets_by_pcap import canonical_pcap_key


def test_canonical_pcap_key_ignores_split_directory():
    first = {
        "label": "aim",
        "pcap_path": "/root/vpn-app/train_val_split_0/train/aim/00004.pcap",
    }
    second = {
        "label": "aim",
        "pcap_path": "/root/vpn-app/train_val_split_1/val/aim/00004.pcap",
    }
    assert canonical_pcap_key(first) == "aim/00004.pcap"
    assert canonical_pcap_key(first) == canonical_pcap_key(second)
