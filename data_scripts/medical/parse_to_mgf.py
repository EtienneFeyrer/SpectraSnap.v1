import re
import sys

def parse_spectra_from_txt(path):
    spectra = []
    current = None
    reading_peaks = False
    peaks_left = 0

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Start of a new spectrum
            if line.startswith("Name:"):
                if current:
                    spectra.append(current)

                name = line.split("Name:")[1].strip()

                # Extract RT from name: e.g. Unknown (0.00_100.8867m/z)
                rt = None
                m = re.search(r"\(([\d\.]+)_", name)
                if m:
                    rt = float(m.group(1))

                current = {
                    "name": name,
                    "precursor_mz": None,
                    "charge": None,
                    "rt": rt,
                    "peaks": [],
                    "expected_peaks": None
                }
                reading_peaks = False

            elif line.startswith("Charge:"):
                charge_str = line.split(":")[1].strip()
                charge = int(charge_str.replace("+", ""))
                current["charge"] = charge

            elif line.startswith("PrecursorMZ:"):
                current["precursor_mz"] = float(line.split(":")[1].strip())

            elif line.startswith("Num Peaks:"):
                peaks_left = int(line.split(":")[1].strip())
                current["expected_peaks"] = peaks_left
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


def write_mgf(spectra, out_path):
    missing_charge = 0
    missing_precursor = 0
    empty_peaks = 0
    peak_mismatch = 0
    removed_second_last_n = 0
    written = 0
    not_singly_charged = 0

    with open(out_path, "w") as f:
        for spec in spectra:

            # Remove spectra whose second-last character is "n"
            if len(spec["name"]) >= 2 and spec["name"][-2] == "n":
                removed_second_last_n += 1
                continue

            # Validate precursor
            if spec["precursor_mz"] is None:
                missing_precursor += 1
                continue

            # Validate charge
            charge = spec["charge"]
            if charge is None:
                missing_charge += 1
                charge = 1  # default

            if charge != 1:
                not_singly_charged += 1
                continue
        
            # Validate peaks
            if not spec["peaks"]:
                empty_peaks += 1
                continue
        
            # Validate peak count mismatch
            if spec["expected_peaks"] is not None:
                if len(spec["peaks"]) != spec["expected_peaks"]:
                    peak_mismatch += 1

            # Write valid spectrum
            f.write("BEGIN IONS\n")
            f.write(f"TITLE={spec['name']}\n")
            f.write(f"PEPMASS={spec['precursor_mz']}\n")

            if charge > 0:
                f.write(f"CHARGE={charge}+\n")
            else:
                f.write(f"CHARGE={abs(charge)}-\n")

            if spec["rt"] is not None:
                f.write(f"RTINSECONDS={spec['rt']}\n")

            for mz, intensity in spec["peaks"]:
                f.write(f"{mz} {intensity}\n")

            f.write("END IONS\n\n")
            written += 1

    # Print validation summary
    print("=== VALIDATION REPORT ===")
    print(f"Total spectra parsed: {len(spectra)}")
    print(f"Written to MGF: {written}")
    print(f"Removed (TITLE has 'n' as second-last character): {removed_second_last_n}")
    print(f"Missing charge: {missing_charge}")
    print(f"Missing precursor m/z: {missing_precursor}")
    print(f"Empty peak lists: {empty_peaks}")
    print(f"Peak count mismatches: {peak_mismatch}")
    print(f"Not singly charged: {not_singly_charged}")
    print("=========================")


def main():
    if len(sys.argv) != 3:
        print("Usage: python parse_to_mgf.py input.txt output.mgf")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    spectra = parse_spectra_from_txt(input_path)
    write_mgf(spectra, output_path)


if __name__ == "__main__":
    main()
