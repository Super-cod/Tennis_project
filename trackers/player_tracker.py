from ultralytics import YOLO

def PlayerTracker():

    def __init__(self,model_path):
        self.model=YOLO(model_path)

    def detect_frames(self,frame):
        player_detection = []
        for frames in frame:
            player_dict=self.detect_frame(frames)
            player_detection.append(player_dict)
        return player_detection

    def detect_frame(self,frame):
        result=self.model.track(frame,presist=True)
        id_name_dict=result.names
        id_box=result.boxes

        player_dict={}
        for box in result.boxes:
            track_id=int(box.id.tolist()[0])
            result=box.xyxy.tolist()[0]
            object_cls_id=box.cls_id.tolist()[0]
            object_cls_name=id_name_dict[object_cls_id]
            if object_cls_name=="person":
                player_dict[track_id]=result

        return player_dict
