import os

def determine_emotional_status(baseline_angle, top_margin, letter_size,
                               line_spacing, word_spacing, pen_pressure, slant_angle):
    """
    Determine the emotional status from the categorized handwriting features.
    Returns a label (string) and a dictionary of scores.
    """
    scores = {
        "Depressed": 0,
        "Anxious": 0,
        "Happy": 0,
        "Calm": 0,
        "Angry": 0
    }

    # --- Depressed / Sad ---
    # Criteria: descending baseline, small letter size, light pen pressure,
    # and strongly left slanted or irregular slant.
    if baseline_angle == 0:
        scores["Depressed"] += 1
    if letter_size == 1:
        scores["Depressed"] += 1
    if pen_pressure == 1:
        scores["Depressed"] += 1
    if slant_angle in [0, 6]:
        scores["Depressed"] += 1
    # Optionally: wider spacing might indicate isolation
    if line_spacing == 0:
        scores["Depressed"] += 1
    if word_spacing == 0:
        scores["Depressed"] += 1

    # --- Anxious / Tense ---
    # Criteria: narrow top margin, small or medium letter size, light pen pressure,
    # slight left slant, and tight spacing.
    if top_margin == 1:
        scores["Anxious"] += 1
    if letter_size in [1, 2]:
        scores["Anxious"] += 1
    if pen_pressure == 1:
        scores["Anxious"] += 1
    if slant_angle == 1:
        scores["Anxious"] += 1
    if line_spacing == 1:
        scores["Anxious"] += 1
    if word_spacing == 1:
        scores["Anxious"] += 1

    # --- Happy / Energetic ---
    # Criteria: ascending baseline, big letter size, heavy pen pressure,
    # slight to moderate right slant, and moderate spacing.
    if baseline_angle == 1:
        scores["Happy"] += 1
    if letter_size == 0:
        scores["Happy"] += 1
    if pen_pressure == 0:
        scores["Happy"] += 1
    if slant_angle in [2, 3]:
        scores["Happy"] += 1
    if line_spacing == 2:
        scores["Happy"] += 1
    if word_spacing == 2:
        scores["Happy"] += 1

    # --- Calm / Neutral ---
    # Criteria: straight baseline, medium letter size, medium pen pressure,
    # straight slant, and balanced spacing.
    if baseline_angle == 2:
        scores["Calm"] += 1
    if letter_size == 2:
        scores["Calm"] += 1
    if pen_pressure == 2:
        scores["Calm"] += 1
    if slant_angle == 5:
        scores["Calm"] += 1
    if line_spacing == 2:
        scores["Calm"] += 1
    if word_spacing == 2:
        scores["Calm"] += 1

    # --- Angry / Agitated ---
    # Criteria: big letter size, heavy pen pressure, and extremely right slant.
    if letter_size == 0:
        scores["Angry"] += 1
    if pen_pressure == 0:
        scores["Angry"] += 1
    if slant_angle == 4:
        scores["Angry"] += 1

    # Choose the emotion with the highest score
    emotion = max(scores, key=scores.get)
    return emotion, scores


if os.path.isfile("emotion_label_list"):
    print("Error: emotion_label_list already exists.")

elif os.path.isfile("feature_list"):
    print("Info: feature_list found.")

    # Read all lines to determine the total number of samples.
    with open("feature_list", "r") as f:
        all_lines = f.readlines()
    total_lines = len(all_lines)
    print(f"Total samples to process: {total_lines}")

    with open("feature_list", "r") as features, open("emotion_label_list", "a") as labels:
        for idx, line in enumerate(features):
            content = line.split()

            # Extract the categorized features as floats or integers
            baseline_angle = float(content[0])
            top_margin = float(content[1])
            letter_size = float(content[2])
            line_spacing = float(content[3])
            word_spacing = float(content[4])
            pen_pressure = float(content[5])
            slant_angle = float(content[6])
            page_id = content[7]

            # Determine emotional status using our rules
            emotion, score_dict = determine_emotional_status(
                baseline_angle, top_margin, letter_size,
                line_spacing, word_spacing, pen_pressure, slant_angle)

            # Write out the features and the predicted emotional label.
            # Here we write the 7 features, then the emotion, then the page id.
            labels.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t" %
                         (str(baseline_angle), str(top_margin), str(letter_size),
                          str(line_spacing), str(word_spacing), str(pen_pressure), str(slant_angle)))
            labels.write("%s\t" % emotion)
            labels.write("%s" % page_id)
            labels.write("\n")

            # Print progress
            progress = (idx + 1) / total_lines * 100
            print(f"Processed {idx + 1}/{total_lines} samples ({progress:.2f}%)")
    print("Done!")
else:
    print("Error: feature_list file not found.")
