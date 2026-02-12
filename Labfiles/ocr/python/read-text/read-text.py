import os
import sys
from dotenv import load_dotenv
from PIL import Image, ImageDraw

# import namespaces
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


def annotate_lines(image_file, read_result):
    """Draw bounding polygons for detected lines and save as lines.jpg."""
    image = Image.open(image_file)
    draw = ImageDraw.Draw(image)

    if read_result is not None and len(read_result.blocks) > 0:
        for line in read_result.blocks[0].lines:
            # bounding_polygon is a list of points with x/y
            poly = [(p.x, p.y) for p in line.bounding_polygon]
            if len(poly) > 1:
                draw.line(poly + [poly[0]], width=3, fill="red")

    image.save("lines.jpg")


def annotate_words(image_file, read_result):
    """Draw bounding polygons for detected words and save as words.jpg."""
    image = Image.open(image_file)
    draw = ImageDraw.Draw(image)

    if read_result is not None and len(read_result.blocks) > 0:
        for line in read_result.blocks[0].lines:
            for word in line.words:
                poly = [(p.x, p.y) for p in word.bounding_polygon]
                if len(poly) > 1:
                    draw.line(poly + [poly[0]], width=2, fill="lime")

    image.save("words.jpg")


def main():
    # Load configuration settings
    load_dotenv()
    ai_endpoint = os.getenv("AI_SERVICE_ENDPOINT")
    ai_key = os.getenv("AI_SERVICE_KEY")

    # Get image file
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    else:
        image_file = "images/Lincoln.jpg"

    # Authenticate Azure AI Vision client
    cv_client = ImageAnalysisClient(
        endpoint=ai_endpoint,
        credential=AzureKeyCredential(ai_key)
    )

    # Read text in image
    with open(image_file, "rb") as f:
        image_data = f.read()
    print(f"\nReading text in {image_file}")
    result = cv_client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ]
    )

    # Print the text
    if result.read is not None:
        print("\nText:")
        for line in result.read.blocks[0].lines:
            print(f"  {line.text}")

        # Annotate the text in the image
        annotate_lines(image_file, result.read)

        # Find individual words in each line
        print("\nIndividual words:")
        for line in result.read.blocks[0].lines:
            for word in line.words:
                print(f"   {word.text}  (Confidence: {word.confidence:.2f}%)")

        # Annotate the words in the image
        annotate_words(image_file, result.read)


if __name__ == "__main__":
    main()
