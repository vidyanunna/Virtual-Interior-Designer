from flask import Flask, render_template, request, jsonify, send_file
from pyngrok import ngrok
import os
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)

# Set your ngrok authentication token
ngrok.set_auth_token("2tLh4GC1Oky5zpujE0UW1G917y9_5wnAfEK3Mu9dU6zdP899t")

# Load the Stable Diffusion model
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

# Load the Real-ESRGAN model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    model=model,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=True if torch.cuda.is_available() else False
)

# Define room types and prompts
room_types = {
    "living room": {
        "modern": "A modern living room with a sofa, coffee table, and large windows, realistic lighting, 4k",
        "minimalist": "A minimalist living room with clean lines, neutral colors, and simple furniture, 4k",
        "vintage": "A vintage living room with antique furniture, patterned rugs, and classic decor, 4k",
        "rustic": "A rustic living room with wooden beams, cozy fireplace, and natural textures, 4k",
        "industrial": "An industrial living room with exposed brick walls, metal accents, and minimalist decor, 4k",
        "bohemian": "A bohemian living room with vibrant colors, eclectic furniture, and layered textiles, 4k",
        "scandinavian": "A Scandinavian living room with light wood, pastel colors, and cozy textiles, 4k",
        "art_deco": "An Art Deco living room with luxurious furniture, geometric patterns, and gold accents, 4k"
    },
    "bedroom": {
        "modern": "A modern bedroom with a king-sized bed, sleek furniture, and large windows, realistic lighting, 4k",
        "minimalist": "A minimalist bedroom with a simple bed, neutral colors, and clean lines, 4k",
        "vintage": "A vintage bedroom with antique furniture, floral patterns, and classic decor, 4k",
        "rustic": "A rustic bedroom with wooden furniture, cozy bedding, and natural textures, 4k",
        "industrial": "An industrial bedroom with exposed brick walls, metal bed frame, and minimalist decor, 4k",
        "bohemian": "A bohemian bedroom with vibrant colors, eclectic furniture, and layered textiles, 4k",
        "scandinavian": "A Scandinavian bedroom with light wood furniture, pastel colors, and cozy bedding, 4k",
        "art_deco": "An Art Deco bedroom with luxurious bedding, geometric patterns, and gold accents, 4k"
    },
    "bathroom": {
        "modern": "A modern bathroom with a sleek bathtub, glass shower, and contemporary fixtures, 4k",
        "minimalist": "A minimalist bathroom with clean lines, neutral colors, and simple fixtures, 4k",
        "vintage": "A vintage bathroom with antique fixtures, patterned tiles, and classic decor, 4k",
        "rustic": "A rustic bathroom with wooden accents, stone tiles, and natural textures, 4k",
        "industrial": "An industrial bathroom with exposed pipes, concrete walls, and minimalist fixtures, 4k",
        "bohemian": "A bohemian bathroom with vibrant tiles, eclectic decor, and layered textiles, 4k",
        "scandinavian": "A Scandinavian bathroom with light wood, white tiles, and simple decor, 4k",
        "art_deco": "An Art Deco bathroom with geometric tiles, luxurious fixtures, and gold accents, 4k"
    },
    "kitchen": {
        "modern": "A modern kitchen with stainless steel appliances, sleek cabinets, and an island, 4k",
        "minimalist": "A minimalist kitchen with clean lines, neutral colors, and simple cabinets, 4k",
        "vintage": "A vintage kitchen with antique cabinets, patterned tiles, and classic decor, 4k",
        "rustic": "A rustic kitchen with wooden cabinets, stone countertops, and natural textures, 4k",
        "industrial": "An industrial kitchen with exposed brick walls, metal accents, and minimalist decor, 4k",
        "bohemian": "A bohemian kitchen with vibrant colors, eclectic decor, and layered textiles, 4k",
        "scandinavian": "A Scandinavian kitchen with light wood, white cabinets, and simple decor, 4k",
        "art_deco": "An Art Deco kitchen with geometric patterns, luxurious finishes, and gold accents, 4k"
    }
}

# Ensure uploads and static directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Global variables to store input image path, style, and room type
last_input_image_path = None
last_style = None
last_room_type = None

# Image transformation function
def transform_room(image_path, prompt, strength=0.75, guidance_scale=7.5):
    init_image = Image.open(image_path).convert("RGB")
    init_image = init_image.resize((768, 768))  # Resize for Stable Diffusion
    with torch.autocast("cuda"):
        # Add randomness with a seed
        generator = torch.Generator(device="cuda").manual_seed(torch.randint(0, 1000000, (1,)).item())
        image = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
    # Enhance the image using Real-ESRGAN
    enhanced_image, _ = upsampler.enhance(np.array(image), outscale=4)
    enhanced_image = Image.fromarray(enhanced_image)
    return enhanced_image

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/living-room')
def living_room():
    return render_template('living-room.html')

@app.route('/bedroom')
def bedroom():
    return render_template('bedroom.html')

@app.route('/bathroom')
def bathroom():
    return render_template('bathroom.html')

@app.route('/kitchen')
def kitchen():
    return render_template('kitchen.html')

@app.route('/transform', methods=['POST'])
def transform():
    global last_input_image_path, last_style, last_room_type

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    style = request.form.get('style')
    room_type = request.form.get('room_type')

    if not style or not room_type:
        return jsonify({"error": "Style or room type not provided"}), 400

    if room_type not in room_types or style not in room_types[room_type]:
        return jsonify({"error": "Invalid room type or style"}), 400

    prompt = room_types[room_type][style]
    input_image_path = os.path.join("uploads", image_file.filename)
    image_file.save(input_image_path)

    # Store globally for re-transformation
    last_input_image_path = input_image_path
    last_style = style
    last_room_type = room_type

    try:
        output_image = transform_room(input_image_path, prompt)
        output_image_path = os.path.join("static", "output.png")
        output_image.save(output_image_path)

        # Save input image to static folder for display
        input_image_static_path = os.path.join("static", "input.png")
        Image.open(input_image_path).save(input_image_static_path)

        logging.info(f"Transformed image saved at {output_image_path}")
        return jsonify({
            "image_url": "/static/output.png",
            "input_image_url": "/static/input.png",
            "style": style,
            "room_type": room_type
        })
    except Exception as e:
        logging.error(f"Error in transform: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/retransform', methods=['POST'])
def retransform():
    global last_input_image_path, last_style, last_room_type

    if not last_input_image_path or not last_style or not last_room_type:
        logging.error("No previous transformation data available")
        return jsonify({"error": "No previous transformation data available"}), 400

    prompt = room_types[last_room_type][last_style]
    logging.info(f"Retrnasforming with prompt: {prompt}, image: {last_input_image_path}")

    try:
        output_image = transform_room(last_input_image_path, prompt)
        output_image_path = os.path.join("static", "output.png")
        output_image.save(output_image_path)
        logging.info(f"Retransformed image saved at {output_image_path}")
        return jsonify({"image_url": "/static/output.png?cache_bust=" + str(torch.randint(0, 1000000, (1,)).item())})
    except Exception as e:
        logging.error(f"Error in retransform: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/result')
def result():
    return render_template('result.html',
                          input_image_url="/static/input.png",
                          output_image_url="/static/output.png",
                          style=last_style,
                          room_type=last_room_type)

@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

# Run the app
if __name__ == '__main__':
    # Start ngrok tunnel
    public_url = ngrok.connect(5000).public_url
    print(f" * ngrok tunnel: {public_url}")

    # Run the Flask app
    app.run(port=5000)