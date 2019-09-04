package org.fxapps.ml.deeplearning.yolo1;

import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.util.HashMap;

public class Yolo2Test {
    public static void main(String[] args) throws Exception {
        System.out.println("loading model ...");
        long loadStart = System.currentTimeMillis();
        Yolo2Prediction yolo2 = Yolo2Prediction.getInstance();
        long loadEnd = System.currentTimeMillis();
        System.out.println("load model cost: "+(loadEnd-loadStart)/1000+"s");
        // 绘图用
        CanvasFrame frame = new CanvasFrame("test");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        String files[] = {"D:/tmp/yolo/cat_416.jpg"};
        for (int i = 0; i < files.length; i++) {
            Mat img = opencv_imgcodecs.imread(files[i]);
            long preStart = System.currentTimeMillis();
            HashMap<String, Integer> countMap = yolo2.objectDetect(img);
            long preEnd = System.currentTimeMillis();
            System.out.println("predict cost: "+(preEnd-preStart)/1000+"s");
            System.out.println(countMap);
        frame.setTitle("test");
        frame.setCanvasSize(416, 416);
        frame.showImage(converter.convert(img));
        frame.waitKey();
        }
    }
}