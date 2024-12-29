import matplotlib.pyplot as plt
import streamlit as st
import torchvision
import builtins
import logging
import mlflow
import base64
import random
import torch
import time
import os
from PIL import Image
from torchvision import transforms
from typing import List, Optional, Tuple


torch.serialization.add_safe_globals([torchvision.transforms._presets.ImageClassification, torchvision.transforms.functional.InterpolationMode, torchvision.datasets.folder.ImageFolder, torchvision.datasets.vision.StandardTransform, torchvision.datasets.folder.default_loader, builtins.set])


# Setup logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Setup page.
about = """This is a basic Image Classification model used to identify 15 different animals. The model leverages *EfficientNet_B0* pytorch model trained on 1944 images of 15 distinct animals *[Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra]*. This app is build in association with *Unified Mentor* for machine learning project submition."""

st.set_page_config(page_title="Fine-Tuned Animal Classifier",
                   page_icon="🐱🦓🐯", menu_items={"About": f"{about}"})
st.title(body="Trouble Identifying Animals 🐱🦓🐯? Here's Animal Classifier to Your Help 👇")
st.markdown("*Identify 15 different animals using this Fine-Tuned Image Classification model. Upload an image or select from test images to get started.*")


# Initialize session state
if 'effnet_transform' not in st.session_state:
    st.session_state.effnet_transform = None
if 'pt_model' not in st.session_state:
    st.session_state.pt_model = None
if 'test_images' not in st.session_state:
    st.session_state.test_images = None
if 'test_img' not in st.session_state:
    st.session_state.test_img = None
if 'selected_test_img' not in st.session_state:
    st.session_state.selected_test_img = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None


# Setting mlflow tracking uri.
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))


# Model loading function.
@st.cache_resource
def load_model(model_name: str) -> Optional[torch.nn.Module]:
    try:
        model_uri = f"models:/{model_name}@champion"
        pt_model = mlflow.pytorch.load_model(model_uri=model_uri, map_location=torch.device('cpu'))
        return pt_model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f":red[Failed to load *{model_name}* model. Please try again later.]", icon="🚨")
        return None


# EfficientNet transform loading function.
@st.cache_data
def load_artifact_(artifact_path: str) -> Optional[torch.Tensor]:
    try:
        artifact_path_ = mlflow.artifacts.download_artifacts(artifact_path=artifact_path.split('/')[-1], run_id=artifact_path.split('/')[-3], tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"))
        artifact_ = torch.load(artifact_path_, weights_only=True)
        return artifact_
    except Exception as e:
        logger.error(f"Error loading artifact: {str(e)}")
        st.error(f":red[Failed to load artifact. Please try again later.]", icon="🚨")
        return None


# Test images loading function.
@st.cache_data
def gather_test_imgs(root_directory: str) -> List[Image.Image]:
    test_images = []
    try:
        for root, dirs, files in os.walk(root_directory):
            for file in files:
                full_path = os.path.join(root, file)
                logger.info(f"Found file: {full_path}")
                test_images.append(full_path)
        return test_images
    except Exception as e:
        logger.error(f"Error gathering test images: {str(e)}")
        st.error(f":red[Failed to gather test images. Please try again later.]", icon="🚨")
        return None


# Profile image loader.
def get_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


# Load model.
if 'pt_model' not in st.session_state or st.session_state.pt_model is None:
    with st.spinner(f":green[This may take a while... Loading *PyTorch Animal Classifier* model... ]"):
        st.session_state.pt_model = load_model(model_name="animal_clf_pytorch")


# Load efficientnet transform
if 'effnet_transform' not in st.session_state or st.session_state.effnet_transform is None:
    with st.spinner(f":green[This may take a while... Loading *EfficientNet_B0 Transform*...]"):
        st.session_state.effnet_transform = load_artifact_("mlflow-artifacts:/719955ada7ef4af9a732dca20a9e381d/5a88f1a67e9f4c409e066a3b1f4f4421/artifacts/effnetb0_transform.pt")


# Load test images
if 'test_images' not in st.session_state or st.session_state.test_images is None:
    with st.spinner(f":green[This may take a while... Loading test images...]"):
        st.session_state.test_images = gather_test_imgs(root_directory="./assets/imgs")


# Load class names
if 'class_names' not in st.session_state or st.session_state.class_names is None:
    with st.spinner(f":green[This may take a while... Loading test images...]"):
        st.session_state.class_names = load_artifact_("mlflow-artifacts:/719955ada7ef4af9a732dca20a9e381d/5a88f1a67e9f4c409e066a3b1f4f4421/artifacts/class_names.pt")


## Main Function ##
def exec_time(start_time: float, end_time: float) -> None:
    total_time = end_time - start_time

    # Break down into hours, minutes, seconds, milliseconds, and microseconds
    hour = int(total_time // 3600)
    minute = int((total_time % 3600) // 60)
    second = int(total_time % 60)
    millisecond = int((total_time * 1000) % 1000)
    microsecond = int((total_time * 1_000_000) % 1000)

    # Construct output dynamically
    components = []
    if hour > 0:
        components.append(f"{hour:02d} hr")
    if minute > 0:
        components.append(f"{minute:02d} min")
    if second > 0:
        components.append(f"{second:02d} sec")
    if millisecond > 0:
        components.append(f"{millisecond:03d} ms")
    if microsecond > 0:
        components.append(f"{microsecond:03d} µs")

    # Display execution time
    if components:
        st.markdown(f":blue[{'took ' + ' : '.join(components)}]")
    else:
        st.markdown("Execution time: < 1 µs")


# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_: str,
                        animal: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: transforms = None) -> None:

    # 2. Open image
    img = Image.open(image_)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ###

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    start_time = time.perf_counter()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image)
    end_time = time.perf_counter()

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    target_image_pred_label = target_image_pred_label.item()

    # 10. Plot image with predicted label and probability
    if isinstance(image_, str):
        true_class = image_.split('/')[-1].split('_')[0]
    else:
        true_class = animal

    pred_class = class_names[target_image_pred_label]
    plt.figure()

    st.markdown("""---""")
    st.header("Classification Result ✨✨✨")
    
    vowels = ['a', 'e', 'i', 'o', 'u']
    article = 'an' if pred_class[0].lower() in vowels else 'a'
    
    st.markdown(f"#### *This is an image of {article} **{pred_class}***.")
    exec_time(start_time, end_time)

    plt.imshow(img)
    
    if true_class == pred_class:
        plt.title(f"True: {true_class} | Pred: {pred_class} | Prob: {target_image_pred_probs.max():.3f}", c='g')
    else:
        plt.title(f"True: {true_class} | Pred: {pred_class} | Prob: {target_image_pred_probs.max():.3f}", c='r')
    
    plt.axis(False);

    # Use Streamlit to display the plot
    st.pyplot(plt)


def display_test_images_grid(test_images_list: List[str]) -> None:
    # Ensure images are persistent across reruns
    if "current_image_grid" not in st.session_state:
        st.session_state.current_image_grid = random.sample(test_images_list, k=16)

    # Create columns for the grid
    num_cols = 4
    cols = st.columns(num_cols)

    # Display images in a grid
    for idx, img_path in enumerate(st.session_state.current_image_grid):
        col = cols[idx % num_cols]
        try:
            img = Image.open(img_path)
            # Create unique key for each image button
            btn_key = f"btn_{idx}"
            
            # Show image and button in the column
            with col:
                st.image(img, use_container_width=True)
                if st.button("Select", key=btn_key):
                    st.session_state.selected_test_img = img_path
                    logger.info(f"Selected test image: {img_path}")
                    
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            st.error(f"Error loading image {img_path}: {e}")
            return None

    if st.session_state.selected_test_img is None:
        st.session_state.selected_test_img = random.sample(test_images_list, k=1)[0]

    # return test_img


# Image Selection Column
st.markdown("""---""")
upload_cols = st.columns([2, 1])
with upload_cols[0]:
    st.markdown("<h1 align='center'>Upload an Image 📷</h1>", unsafe_allow_html=True)

    # Option 1: File uploader
    help_text = """Consider uploading colorful RGB image. Uploading grayscale image may lead to error while prediction."""

    uploaded_img = st.file_uploader("Choose a png or jpg file", type=["png", "jpg"], help=f":blue[{help_text}]")

    # Dropdown menu for selecting ASL alphabets
    selected_animal = st.selectbox("Choose an Animal uploaded image represents", st.session_state.class_names)

    st.markdown("<h2 align='center'>OR</h2>", unsafe_allow_html=True)

    # Option 2: Test image selection
    with st.expander("Select from Test Images"):
        st.markdown(":blue[**Hint:** Remove uploaded image, if any, before selecting from test images.]")

        display_test_images_grid(st.session_state.test_images)


# # Update the current image based on upload or selection
if uploaded_img:
    st.session_state.test_img = uploaded_img
    st.session_state.selected_test_image = None
elif st.session_state.selected_test_img:
    st.session_state.test_img = st.session_state.selected_test_img

with upload_cols[1]:
    try:
        if st.session_state.test_img:
            st.markdown("<h5 align='center' style='color: green'>Selected Image</h5>", unsafe_allow_html=True)
            display_img = Image.open(st.session_state.test_img)
            st.image(display_img, use_container_width=True)
    except Exception as e:  
        logger.error(f"Error processing uploaded image: {str(e)}")
        st.error("Failed to process the uploaded image. Please try again with a different image.")

classify_btn = upload_cols[0].button(":red[Detect Sign]")

# classify_btn click event
if classify_btn:
    with st.spinner(":blue[Classifying...]"):
        pred_and_plot_image(st.session_state.pt_model, st.session_state.test_img, selected_animal, st.session_state.class_names, (244, 244), st.session_state.effnet_transform)


# Disclamer
st.write("\n"*3)
st.markdown("""----""")
st.write("""*Disclamer: Predictions made by the models may be inaccurate due to the nature of the models and image data. This is a simple demonstration of how machine learning can be used to make predictions. For more accurate predictions, consider using more complex models and larger & diverse dataset.*""")
    
st.markdown("""---""")
st.markdown("Created by [Pranay Jagtap](https://pranayjagtap.netlify.app)")

# Get the base64 string of the image
img_base64 = get_image_base64("assets/pranay_sq.jpg")

# Create the HTML for the circular image
html_code = f"""
<style>
    .circular-image {{
        width: 125px;
        height: 125px;
        border-radius: 55%;
        overflow: hidden;
        display: inline-block;
    }}
    .circular-image img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}
</style>
<div class="circular-image">
    <img src="data:image/jpeg;base64,{img_base64}" alt="Pranay Jagtap">
</div>
"""

# Display the circular image
st.markdown(html_code, unsafe_allow_html=True)
# st.image("assets/pranay_sq.jpg", width=125)
st.markdown("Electrical Engineer | Machine Learning Enthusiast"\
            "<br>📍 Nagpur, Maharashtra, India", unsafe_allow_html=True)
