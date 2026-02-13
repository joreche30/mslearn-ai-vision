import os
import base64
from openai import AzureOpenAI

def save_base64_image(b64_data: str, output_path: str):
    image_bytes = base64.b64decode(b64_data)
    with open(output_path, "wb") as f:
        f.write(image_bytes)

def main():
    client = AzureOpenAI(
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY")
    )

    # Task 2 — Generate an image from a prompt
    prompt = "A robot holding a cup of coffee"
    result = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024"
    )

    image_base64 = result.data[0].b64_json
    save_base64_image(image_base64, "generated_coffee_robot.png")
    print("Saved: generated_coffee_robot.png")

    # Task 3 — Edit an existing image (inpainting)
    edit_result = client.images.edit(
        model="dall-e-3",
        image=open("city.png", "rb"),
        mask=open("mask.png", "rb"),
        prompt="Add a futuristic flying car in the masked area",
        size="1024x1024"
    )

    edited_base64 = edit_result.data[0].b64_json
    save_base64_image(edited_base64, "edited_city.png")
    print("Saved: edited_city.png")

if __name__ == "__main__":
    main()
