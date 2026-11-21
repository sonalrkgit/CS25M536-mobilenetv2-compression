from torchvision.models import mobilenet_v2

def get_model():
    m = mobilenet_v2(width_mult=1.0, num_classes=10, dropout=0.2)
    m.features[0][0].stride = (1, 1)
    m.features[2].conv[1][0].stride = (1, 1)
    return m
