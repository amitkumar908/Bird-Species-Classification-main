import streamlit as st
from PIL import Image, ImageDraw
import torch
import numpy as np
from torchvision import transforms, models
from ultralytics import YOLO
from transformers import pipeline
import tempfile


# ─── 0)class NAMES───_____─────────────
class_names = [
    "Barn Swallow", "Bay breasted Warbler", "Black and white Warbler", "Black billed Cuckoo",
    "Black throated Blue Warbler", "Black throated Sparrow", "Blue Grosbeak", "Blue Jay",
    "Bobolink", "Bohemian Waxwing", "Bronzed Cowbird", "Brown Creeper",
    "Brown Pelican", "Brown Thrasher", "Canada Warbler", "Cape Glossy Starling",
    "Cape May Warbler", "Cardinal", "Carolina Wren", "Caspian Tern",
    "Cedar Waxwing", "Cerulean Warbler", "Chuck will Widow", "Clark Nutcracker",
    "Common Yellowthroat", "Eared Grebe", "Eastern Towhee", "European Goldfinch",
    "Evening Grosbeak", "Forsters Tern", "Fox Sparrow", "Geococcyx",
    "Golden winged Warbler", "Gray Kingbird", "Gray crowned Rosy Finch", "Green Jay",
    "Green Violetear", "Green tailed Towhee", "Harris Sparrow", "Heermann Gull",
    "Hooded Merganser", "Hooded Oriole", "Hooded Warbler", "Horned Grebe",
    "Horned Lark", "Horned Puffin", "Ivory Gull", "Lazuli Bunting",
    "Le Conte Sparrow", "Least Auklet", "Loggerhead Shrike", "Magnolia Warbler",
    "Mallard", "Myrtle Warbler", "Nashville Warbler", "Nelson Sharp tailed Sparrow",
    "Nighthawk", "Ovenbird", "Pacific Loon", "Painted Bunting",
    "Palm Warbler", "Parakeet Auklet", "Pied Kingfisher", "Pied billed Grebe",
    "Pine Grosbeak", "Pine Warbler", "Prairie Warbler", "Prothonotary Warbler",
    "Purple Finch", "Red bellied Woodpecker", "Red cockaded Woodpecker", "Red eyed Vireo",
    "Red winged Blackbird", "Rhinoceros Auklet", "Rose breasted Grosbeak", "Ruby throated Hummingbird",
    "Rufous Hummingbird", "Savannah Sparrow", "Sayornis", "Scarlet Tanager",
    "Scissor tailed Flycatcher", "Spotted Catbird", "Summer Tanager", "Tree Swallow",
    "Tropical Kingbird", "Vermilion Flycatcher", "Vesper Sparrow", "Warbling Vireo",
    "Western Meadowlark", "Western Wood Pewee", "Whip poor Will", "White Pelican",
    "White breasted Kingfisher", "White breasted Nuthatch", "White crowned Sparrow", "White throated Sparrow",
    "Yellow Warbler", "Yellow breasted Chat", "Yellow headed Blackbird", "Yellow throated Vireo"
]

# ─── 1) CACHING MODEL LOADERS ──────────────────────────────────────────────

@st.cache_resource
def load_yolo_model(path="models/epoch45.pt"):
    return YOLO(path)

@st.cache_resource
def load_resnet_model(path="models/model_state_dict_best.pth", device="cpu"):
    # 1) Build the bare ResNet152
    model = models.resnet152(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

    # 2) Load the raw state_dict
    raw_sd = torch.load(path, map_location="cpu")

    # 3) Strip "module." prefixes if present
    new_sd = {}
    for k, v in raw_sd.items():
        name = k
        if k.startswith("module."):
            name = k[len("module."):]
        new_sd[name] = v

    # 4) LOAD INTO THE MODEL
    model.load_state_dict(new_sd)

    # 5) Wrap for multi‑GPU (if you’ll run on GPU) and move to device
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_swin_pipeline():
    return pipeline("image-classification", model="Emiel/cub-200-bird-classifier-swin")

# ─── 2) TRANSFORM FOR RESNET ─────────────────────────────────────────────────____________________________

resnet_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ─── 3) STREAMLIT UI ────────────────────────────────────────────────────────

st.title("🐦 Bird Species Classification")

uploaded = st.file_uploader("Upload a bird image", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Please upload an image.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Uploaded Image", use_container_width=True)

# Load models
yolo_model       = load_yolo_model("models/epoch45.pt")
resnet_model     = load_resnet_model("models/model_state_dict_best.pth")
hf_pipe          = load_swin_pipeline()

# Run YOLO detection
results = yolo_model(np.array(img))
boxes   = results[0].boxes.xyxy.cpu().numpy()

annotated = img.copy()
draw = ImageDraw.Draw(annotated)

# Precompute HF result on the full image
hf_res_full = hf_pipe(img)
hf_label, hf_conf = hf_res_full[0]["label"], hf_res_full[0]["score"]

for box in boxes:
    x1,y1,x2,y2 = map(int, box)
    crop = img.crop((x1,y1,x2,y2))

    # ResNet prediction on the crop
    inp = resnet_transform(crop).unsqueeze(0)
    with torch.no_grad():
        out   = resnet_model(inp)
        probs = torch.softmax(out, dim=1)[0]
        ridx  = torch.argmax(probs).item()
        rlabel, rconf = class_names[ridx], probs[ridx].item()

    # Choose best
    if hf_conf > rconf:
        label, score, method = hf_label, hf_conf, "HF"
    else:
        label, score, method = rlabel, rconf, "ResNet"

    # Draw box + label
    draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
    draw.text((x1, max(0,y1-15)),
              f"{label} ({score:.2f})",
              fill="red")

st.image(annotated, caption="Detections & Predictions", use_container_width=True)
