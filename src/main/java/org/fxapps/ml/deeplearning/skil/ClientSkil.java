package org.fxapps.ml.deeplearning.skil;

import ai.skymind.ApiClient;
import ai.skymind.auth.ApiKeyAuth;
import ai.skymind.skil.DefaultApi;
import ai.skymind.skil.model.INDArray;
import ai.skymind.skil.model.LoginRequest;
import ai.skymind.skil.model.LoginResponse;
import ai.skymind.skil.model.Prediction;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.zoo.util.ClassPrediction;
import org.deeplearning4j.zoo.util.darknet.COCOLabels;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;

import java.io.File;
import java.util.HashMap;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

public class ClientSkil {
    public static void main(String[] args) {

        ApiClient apiClient = new ApiClient();
        apiClient.setBasePath("http://192.168.1.128:9008"); // Replace this with the host and port of your SKIL server, if required.

        DefaultApi apiInstance = new DefaultApi(apiClient);


        try {
            LoginResponse loginResponse = apiInstance.login(
                    new LoginRequest()
                            .userId("admin")
                            .password("123456")
            );

            ApiKeyAuth api_key = (ApiKeyAuth) apiClient.getAuthentication("api_key");
            api_key.setApiKeyPrefix("Bearer");
            api_key.setApiKey(loginResponse.getToken());
            String file = "D:/tmp/yolo/cat_416.jpg";
            Prediction result = apiInstance.predictimage("age", "default", "javayolov2", new File(file));
            org.nd4j.linalg.api.ndarray.INDArray nd = Nd4jBase64.fromBase64(result.getPrediction().getArray());

            double[][] priorBoxes = {{2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}};
            List<DetectedObject> objs = YoloUtils.getPredictedObjects((Nd4j.create(priorBoxes)), nd, 0.5, 0.3);
            COCOLabels labels = new COCOLabels();

            // 固定值
            int width = 416;
            int height = 416;
            int gridWidth = 13;
            int gridHeight = 13;
            opencv_core.Mat img = opencv_imgcodecs.imread(file);
            int w = img.arrayWidth();
            int h = img.arrayHeight();

            CanvasFrame frame = new CanvasFrame("test");
            OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
            HashMap<String, Integer> countMap = new HashMap<>();
            Integer value = 1;

            for (DetectedObject obj : objs) {
//                System.out.print(obj.getClassPredictions());
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
                putText(img, label, new opencv_core.Point(x1 + 2, y2 - 2), 2, 1, opencv_core.Scalar.GREEN);

                System.out.println("label =>"+label);
                // 判断字典
                if(countMap!=null && countMap.containsKey(label)){
                    value = countMap.get(label) + 1;
                }
                countMap.put(label, value);
            }
//            JavaCVUtil.imShow(img,"处理好的图片");
            System.out.println(countMap);
            frame.setTitle("test");
            frame.setCanvasSize(416, 416);
            frame.showImage(converter.convert(img));
            frame.waitKey();
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
