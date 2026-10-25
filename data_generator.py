
import os
import numpy as np
import pandas as pd


def generate_network_data(samples_per_class=2500, random_state=42):
    """Generate reproducible synthetic network traffic.
    This is simulated educational data, not real network traffic.
    """
    rng = np.random.default_rng(random_state)

    profiles = {
        "Normal": {
            "duration": (1.8, 1.0),
            "packets": (75, 45),
            "bytes_per_packet": (350, 180),
            "failed_connections": (1.5, 2.0),
            "connection_rate": (45, 35),
        },
        "DoS": {
            "duration": (0.9, 0.8),
            "packets": (280, 170),
            "bytes_per_packet": (180, 100),
            "failed_connections": (2.0, 2.5),
            "connection_rate": (260, 180),
        },
        "Brute Force": {
            "duration": (1.3, 0.9),
            "packets": (55, 35),
            "bytes_per_packet": (130, 75),
            "failed_connections": (8, 6),
            "connection_rate": (70, 50),
        },
        "Port Scan": {
            "duration": (0.7, 0.6),
            "packets": (22, 16),
            "bytes_per_packet": (100, 65),
            "failed_connections": (5, 5),
            "connection_rate": (160, 110),
        },
        "Data Exfiltration": {
            "duration": (4.0, 2.5),
            "packets": (190, 120),
            "bytes_per_packet": (520, 280),
            "failed_connections": (1.5, 2.0),
            "connection_rate": (55, 40),
        },
    }

    frames = []

    for label, profile in profiles.items():
        n = samples_per_class

        duration = np.clip(
            rng.normal(*profile["duration"], n), 0.01, None
        )
        packets = np.clip(
            np.rint(rng.normal(*profile["packets"], n)), 1, None
        ).astype(int)
        bytes_per_packet_sample = np.clip(
            rng.normal(*profile["bytes_per_packet"], n), 20, None
        )
        failed = np.clip(
            np.rint(rng.normal(*profile["failed_connections"], n)),
            0, None
        ).astype(int)
        rate = np.clip(
            rng.normal(*profile["connection_rate"], n), 0.1, None
        )

        byte_count = np.maximum(
            np.rint(packets * bytes_per_packet_sample), 1
        ).astype(int)

        # Engineered features
        failed_connection_ratio = failed / np.maximum(packets, 1)
        traffic_intensity = byte_count / duration

        frame = pd.DataFrame({
            "duration": duration,
            "packets": packets,
            "bytes": byte_count,
            "failed_connections": failed,
            "connection_rate": rate,
            "bytes_per_packet": byte_count / packets,
            "failed_connection_ratio": failed_connection_ratio,
            "traffic_intensity": traffic_intensity,
            "label": label,
        })

        frames.append(frame)

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    return dataset


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = generate_network_data(
        samples_per_class=2500,
        random_state=42,
    )

    output_path = "data/network_traffic.csv"
    df.to_csv(output_path, index=False)

    print("Synthetic dataset generated successfully.")
    print("Shape:", df.shape)
    print("\nClass distribution:")
    print(df["label"].value_counts())
    print("\nMissing values:", int(df.isnull().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("Saved to:", output_path)

