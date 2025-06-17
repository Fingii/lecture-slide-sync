from slide_detection import detect_slide_transitions

if __name__ == "__main__":
    detect_slide_transitions(
        "tests/test_data/videos/dbwt1/dbwt1_02.mp4", "tests/test_data/slides_pdf/dbwt1/dbwt1_02.pdf"
    )
