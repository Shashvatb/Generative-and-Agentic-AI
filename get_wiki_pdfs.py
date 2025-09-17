"""
This code is AI generated in order to scrape data from wikipedia for us to use
"""

import wikipedia
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def save_wikipedia_to_pdf(topic: str):
    """Fetch a Wikipedia page and save it as a PDF."""
    try:
        # Get page (avoids disambiguation issues)
        page = wikipedia.page(topic, auto_suggest=False)
        content = page.content

        # Create PDF path
        pdf_path = os.path.join(f"{topic.replace(' ', '_')}.pdf")

        # Create PDF
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 72, topic)

        # Content
        c.setFont("Helvetica", 12)
        y = height - 100
        for line in content.split("\n"):
            for chunk in [line[i:i+90] for i in range(0, len(line), 90)]:
                c.drawString(72, y, chunk)
                y -= 15
                if y < 72:  # new page if near bottom
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y = height - 72

        c.save()
        print(f"Saved {topic} → {pdf_path}")

    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation for {topic}, picking first option: {e.options[0]}")
        save_wikipedia_to_pdf(e.options[0])
    except Exception as e:
        print(f"Error fetching {topic}: {e}")


if __name__ == "__main__":
    starters = ["Bulbasaur", "Charmander", "Squirtle"]

    for s in starters:
        save_wikipedia_to_pdf(s)
