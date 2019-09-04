package org.fxapps.ml.deeplearning.skil;

//import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_DUPLEX;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;

import java.util.List;

import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

public class Yolo2Utils {
	
	//要检测的标签，要按实际情况修改
		public static final String[] CLASSES = {"aeroplane","bicycle", "bird" , "boat" ,"bottle" , "bus" , "car" ,
                "cat" , "chair" , "cow", "diningtable" , "dog" , "horse" , "motorbike" ,
                "person" , "pottedplant", "sheep" , "sofa" , "train", "tvmonitor"};
	
		
	 	
	
	 public static  void markWithBoundingBox(Mat file, int gridWidth, int gridHeight, int w, int h, List<DetectedObject> objects) {

		   for(DetectedObject obj:objects) {
	        double[] xy1 = obj.getTopLeftXY();
	        double[] xy2 = obj.getBottomRightXY();
	        int predictedClass = obj.getPredictedClass();
	        int x1 = (int) Math.round(w * xy1[0] / gridWidth);
	        int y1 = (int) Math.round(h * xy1[1] / gridHeight);
	        int x2 = (int) Math.round(w * xy2[0] / gridWidth);
	        int y2 = (int) Math.round(h * xy2[1] / gridHeight);
	        rectangle(file, new Point(x1, y1), new Point(x2, y2), Scalar.GREEN);
	        putText(file, CLASSES[predictedClass], new Point(x1 + 2, y2 - 2), 1, 1, Scalar.GREEN);
		   }	  
      }

}
