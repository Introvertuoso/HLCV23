# https://github.com/salesforce/LAVIS/blob/main/examples/blip_feature_extraction.ipynb

from lavis.models import load_model_and_preprocess, load_model


def get_image_features(raw_image, device, model=None, process=False):
    if process:
        raw_image = preprocess(raw_image, device)
    if model is None:
        model = define_model(device)
    return model.extract_features({"image": raw_image}, mode="image").image_embeds


def define_model(device):
    return load_model(
        name="blip_feature_extractor",
        model_type="base",
        is_eval=True,
        device=device
    )


def preprocess(raw_image, device):
    return get_transform(device)(raw_image).unsqueeze(0).to(device)


def get_transform(device):
    _, vis_processors, _ = load_model_and_preprocess(
        name="blip_feature_extractor",
        model_type="base",
        is_eval=True,
        device=device
    )
    return vis_processors["eval"].transform
