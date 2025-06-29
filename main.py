from slide_detection import detect_slide_transitions

if __name__ == "__main__":
    keywords = {"UNIVERSITY", "FH", "AACHEN", "OF", "APPLIED", "SCIENCES"}
    detect_slide_transitions(
        "tests/test_data/videos/dbwt1/dbwt1_02.mp4", "tests/test_data/slides_pdf/dbwt1/dbwt1_02.pdf", keywords
    )
