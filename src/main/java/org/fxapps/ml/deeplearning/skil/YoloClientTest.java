package org.fxapps.ml.deeplearning.skil;

import ai.skymind.ApiClient;
import ai.skymind.Configuration;
import ai.skymind.skil.DefaultApi;
import ai.skymind.skil.model.INDArray.OrderingEnum;
import ai.skymind.skil.model.LoginRequest;
import ai.skymind.skil.model.LoginResponse;
import ai.skymind.skil.model.Prediction;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

//import ai.skymind.skil.model.INDArray;

public class YoloClientTest {

    public static void main(String[] args) {

        try {


            ApiClient defaultClient = Configuration.getDefaultApiClient();

            defaultClient.setBasePath("http://192.168.1.128:9008");

            // Configure API key authorization: api_key
            DefaultApi defaultApi = new DefaultApi();
            LoginRequest loginRequest = new LoginRequest();
            loginRequest.setUserId("admin");
            loginRequest.setPassword("123456");
            LoginResponse loginResponse = defaultApi.login(loginRequest);
            String token = loginResponse.getToken();
            defaultClient.setApiKey(token);
            defaultClient.setApiKeyPrefix("Bearer");

            // RGB图
            String fileName = "D:/tmp/yolo/cat_416.jpg";

            File image = new File(fileName);

            Mat imageMat = opencv_imgcodecs.imread(image.getAbsolutePath());


            BufferedImage bufferedImage = ImageIO.read(image);


            Prediction predictionBody = getYolo2PredictionBody(fileName);


            Prediction result = defaultApi.predict(predictionBody, "age", "default", "javayolov2");


            String predict_return_array = result.getPrediction().getArray();

            INDArray boundingBoxPriors = Nd4j.create(YOLO2.DEFAULT_PRIOR_BOXES);

            INDArray networkGlobalOutput = Nd4jBase64.fromBase64(predict_return_array);


            List<DetectedObject> predictedObjects = YoloUtils.getPredictedObjects(boundingBoxPriors, networkGlobalOutput, 0.1, 0.0);


            //排除掉同一个物体的多余的边界框
            predictedObjects = NonMaxSuppression.getObjects(predictedObjects);
            System.out.print(image.getAbsolutePath().replaceAll("JPEGImages", "DetectedJPEGImages"));


            Yolo2Utils.markWithBoundingBox(imageMat, 13, 13, bufferedImage.getWidth(), bufferedImage.getHeight(), predictedObjects);

            opencv_imgcodecs.imwrite(image.getAbsolutePath().replaceAll("JPEGImages", "DetectedJPEGImages"), imageMat);

	        
	        
	        /* MultiPredictResponse multiPredictResponse = skil.multipredictimage(
            imageFile,
            UUID.randomUUID().toString(),
            false,
            "exp1model1", "default", "age"
    );

    assert(multiPredictResponse.getOutputs().size() == 2);
    INDArray array1 = Nd4jBase64.fromBase64(multiPredictResponse.getOutputs().get(0).getArray());
    INDArray array2 = Nd4jBase64.fromBase64(multiPredictResponse.getOutputs().get(1).getArray());

    System.out.println(array1.toString());
    System.out.println(array2.toString());*/

        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }

    @SuppressWarnings("unused")
    private static Prediction getPredictionBody(String imageFile) throws IOException, FileNotFoundException {

        BufferedImage bufferedImage = ImageIO.read(new FileInputStream(imageFile));

        int width = bufferedImage.getWidth();

        int height = bufferedImage.getHeight();

        Float[] pixels = new Float[width * height];

        for (int y = 0; y < height; y++) {

            for (int x = 0; x < width; x++) {

                int color = bufferedImage.getRGB(x, y);

                int red = (color >>> 16) & 0xFF;

                int green = (color >>> 8) & 0xFF;

                int blue = (color >>> 0) & 0xFF;
                //转为单通道灰度值
                float luminance = (red * 0.2126f + green * 0.7152f + blue * 0.0722f) / 255;

                pixels[(y * width) + x] = luminance;

            }
        }

        ai.skymind.skil.model.INDArray pred = new ai.skymind.skil.model.INDArray();

        pred.setOrdering(OrderingEnum.C);

        pred.setData(Arrays.asList(pixels));


        Integer[] shape = {1, width * height};

        pred.setShape(Arrays.asList(shape));

        Prediction body = new Prediction();

        body.setId(String.valueOf(System.currentTimeMillis()));

        body.setNeedsPreProcessing(false);

        body.setPrediction(pred);

        return body;
    }


    private static Prediction getYolo2PredictionBody(String imageFile) throws IOException, FileNotFoundException {

        Mat imageMat = opencv_imgcodecs.imread(imageFile);

        NativeImageLoader loader = new NativeImageLoader(416, 416, 3);

        INDArray features = loader.asMatrix(imageMat);

        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

        scaler.transform(features);

        String featuresString = Nd4jBase64.base64String(features);

        Integer[] shape = {3, 416 * 416};

        ai.skymind.skil.model.INDArray pred = new ai.skymind.skil.model.INDArray();

        pred.setOrdering(OrderingEnum.C);

        pred.setArray(featuresString);

        pred.setShape(Arrays.asList(shape));

        Prediction body = new Prediction();

        body.setId(String.valueOf(System.currentTimeMillis()));

        body.setNeedsPreProcessing(false);

        body.setPrediction(pred);

        return body;
    }


}
