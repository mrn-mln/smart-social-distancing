def build_detection_model(name, config):
    detector_name = name.split("_")[-1]
    if detector_name == "ssd":
        from libs.detectors.x86 import mobilenet_ssd
        detector = mobilenet_ssd.Detector(config=config)
    elif detector_name == "yolov3":
        from libs.detectors.x86 import yolov3
        detector = yolov3.Detector(config=config)
    else:
        raise ValueError('Not supported detector named: ', name, ' for AlphaPose.')
    return detector
