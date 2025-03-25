import os
import categorize

if os.path.isfile("feature_list"):
    print("Error: feature_list already exists.")

elif os.path.isfile("raw_feature_list"):
    print("Info: raw_feature_list found.")

    # Read all lines to determine the total number of samples.
    with open("raw_feature_list", "r") as f:
        all_lines = f.readlines()
    total_lines = len(all_lines)
    print(f"Total samples to process: {total_lines}")

    with open("raw_feature_list", "r") as raw_features, open("feature_list", "a") as features:
        for idx, line in enumerate(raw_features):
            content = line.split()

            raw_baseline_angle = float(content[0])
            raw_top_margin = float(content[1])
            raw_letter_size = float(content[2])
            raw_line_spacing = float(content[3])
            raw_word_spacing = float(content[4])
            raw_pen_pressure = float(content[5])
            raw_slant_angle = float(content[6])
            page_id = content[7]

            baseline_angle, comment = categorize.determine_baseline_angle(raw_baseline_angle)
            top_margin, comment = categorize.determine_top_margin(raw_top_margin)
            letter_size, comment = categorize.determine_letter_size(raw_letter_size)
            line_spacing, comment = categorize.determine_line_spacing(raw_line_spacing)
            word_spacing, comment = categorize.determine_word_spacing(raw_word_spacing)
            pen_pressure, comment = categorize.determine_pen_pressure(raw_pen_pressure)
            slant_angle, comment = categorize.determine_slant_angle(raw_slant_angle)

            features.write("%s\t" % str(baseline_angle))
            features.write("%s\t" % str(top_margin))
            features.write("%s\t" % str(letter_size))
            features.write("%s\t" % str(line_spacing))
            features.write("%s\t" % str(word_spacing))
            features.write("%s\t" % str(pen_pressure))
            features.write("%s\t" % str(slant_angle))
            features.write("%s\t" % str(page_id))
            features.write("\n")

            # Print progress
            progress = (idx + 1) / total_lines * 100
            print(f"Processed {idx + 1}/{total_lines} samples ({progress:.2f}%)")
    print("Done!")

else:
    print("Error: raw_feature_list file not found.")
