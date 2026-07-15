import re
import sys
import json
from pathlib import Path


def parse_spectra_from_txt(path):
    spectra = []
    current = None
    reading_peaks = False
    peaks_left = 0

    for line in open(path, "r"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("Name:"):
            if current:
                spectra.append(current)

            name = line.split("Name:")[1].strip()

            current = {
                "raw_name": name,
                "precursor_mz": None,
                "charge": None,
                "adduct": None,
                "peaks": []
            }
            reading_peaks = False

        elif line.startswith("Charge:"):
            charge_str = line.split(":")[1].strip()
            charge = int(charge_str.replace("+", ""))
            current["charge"] = charge

        elif line.startswith("Precursor_type:"):
            adduct = line.split(":", 1)[1].strip()
            current["adduct"] = adduct

        elif line.startswith("PrecursorMZ:"):
            current["precursor_mz"] = float(line.split(":")[1].strip())

        elif line.startswith("Num Peaks:"):
            peaks_left = int(line.split(":")[1].strip())
            reading_peaks = True

        elif reading_peaks:
            if peaks_left > 0:
                parts = line.split()
                if len(parts) == 2:
                    mz, intensity = parts
                    current["peaks"].append((float(mz), float(intensity)))
                peaks_left -= 1
            if peaks_left == 0:
                reading_peaks = False

    if current:
        spectra.append(current)

    return spectra


def convert_to_massspecgym_json(spectra, out_path, original_filename):
    DEFAULT_INSTRUMENT = "Orbitrap"
    DEFAULT_COLLISION_ENERGY = 50.0
    DEFAULT_SIMULATION_CHALLENGE = False

    output = []
    counter = 1
    skip = 0

    for spec in spectra:

        raw_name = spec["raw_name"]

        # KEEP ONLY spectra whose names end with "n)"
        if not raw_name.endswith("n)"):
            continue

        # Extract retention time + neutral mass
        m = re.search(r"\(([\d\.]+)_([\d\.]+)n\)", raw_name)
        if not m:
            continue

        retention_time = float(m.group(1))
        neutral_mass = float(m.group(2))
        #ID = f"{retention_time}_{neutral_mass}n"
        new_ID = raw_name.replace("Unknown (", "").replace(")", "")
        # Must have peaks
        if not spec["peaks"]:
            continue

        # Normalize intensities
        intensities = [i for _, i in spec["peaks"]]
        max_int = max(intensities)
        if max_int == 0:
            continue

        peaks_json = [
            [mz, intensity / max_int]
            for mz, intensity in spec["peaks"]
        ]

        entry = {
            "identifier": new_ID,
            "retention_time": retention_time,
            "neutral_mass": neutral_mass,
            "precursor_formula": None,  # not available
            "parent_mass": spec["precursor_mz"] - 1.007276 if spec["charge"] else None,
            "precursor_mz": spec["precursor_mz"],
            "peaks_mz": [],
            "adduct": spec["adduct"],
            "instrument_type": DEFAULT_INSTRUMENT,
            "collision_energy": DEFAULT_COLLISION_ENERGY,
            "simulation_challenge": DEFAULT_SIMULATION_CHALLENGE,
            "peaks_json": peaks_json
        }
        # only keep spectra that are lighter than 1000 Da
        if entry["neutral_mass"] > 1000:
            skip += 1
            continue
        output.append(entry)
        counter += 1

    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Reduced {len(spectra)} spectra → {len(output)} spectra after filtering and conversion.")
    print(f"Skipped {skip} spectra with neutral mass > 1000 Da.")


def main():
    if len(sys.argv) != 3:
        print("Usage: python parse_to_json.py input.txt output.json")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = sys.argv[2]

    spectra = parse_spectra_from_txt(input_path)
    convert_to_massspecgym_json(spectra, output_path, input_path.stem)

    #print(f"Converted {len(spectra)} spectra → MassSpecGym JSON: {output_path}")


if __name__ == "__main__":
    main()
