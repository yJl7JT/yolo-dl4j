package org.fxapps.ml.deeplearning.yolo1;

import org.bytedeco.javacpp.opencv_core;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.util.ClassPrediction;
import org.deeplearning4j.zoo.util.darknet.COCOLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import java.io.File;
import java.util.HashMap;
import java.util.List;
import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_DUPLEX;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;


public class Yolo2Prediction {
    private ComputationGraph initializedModel;
    //使用volatile关键字保其可见性
    volatile private static Yolo2Prediction instance = null;

    private Yolo2Prediction() throws Exception {
        //"http://blob.deeplearning4j.org/models/yolo2_dl4j_inference.v3.zip"
        File locationToSave = new File("D:\\model\\yolo\\yolo2\\yolo2_dl4j_inference.v3.zip");
        initializedModel = ModelSerializer.restoreComputationGraph(locationToSave);

    }

    public static Yolo2Prediction getInstance() throws Exception {

        if(instance == null){
            synchronized (Yolo2Prediction.class) {
                if(instance == null){//二次检查
                    instance = new Yolo2Prediction();
                }
            }
        }
        return instance;
    }

    // 多线程下线程不安全
    synchronized public HashMap<String, Integer> objectDetect(opencv_core.Mat img) throws Exception {
        // 字典存储类别及其数目
        HashMap<String, Integer> countMap = new HashMap<>();
        int w = img.arrayWidth();
        int h = img.arrayHeight();
        // 固定值
        int width = 416;
        int height = 416;
        int gridWidth = 13;
        int gridHeight = 13;

        // 转为ndarray
        NativeImageLoader loader = new NativeImageLoader(height, width, 3);
        System.out.println(img);
        INDArray image = loader.asMatrix(img);

        // 缩放image
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);

        // 预测结果，image仅在此用
        System.out.println(image);
        INDArray outputs = initializedModel.outputSingle(image);
        double[][] priorBoxes = { { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 } };
        List<DetectedObject> objs = YoloUtils.getPredictedObjects((Nd4j.create(priorBoxes)), outputs, 0.5, 0.3);

        // 标签数据
        COCOLabels labels = new COCOLabels();
        Integer value = 1;
        for (DetectedObject obj : objs) {

            // 预测类别
            ClassPrediction classPrediction = labels.decodePredictions(obj.getClassPredictions(), 1).get(0).get(0);
            String label = classPrediction.getLabel();

            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();

            // 矩形框四个点坐标
            int x1 = (int) Math.round(w * xy1[0] / gridWidth);
            int y1 = (int) Math.round(h * xy1[1] / gridHeight);
            int x2 = (int) Math.round(w * xy2[0] / gridWidth);
            int y2 = (int) Math.round(h * xy2[1] / gridHeight);

            // 矩形框-红色
            rectangle(img, new opencv_core.Point(x1, y1), new opencv_core.Point(x2, y2), opencv_core.Scalar.RED);
            // 文字-绿色
            putText(img, label, new opencv_core.Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, opencv_core.Scalar.GREEN);

            // 判断字典
            if(countMap!=null && countMap.containsKey(label)){
                value = countMap.get(label) + 1;
            }
            countMap.put(label, value);
        }
        return countMap;
    }
}
