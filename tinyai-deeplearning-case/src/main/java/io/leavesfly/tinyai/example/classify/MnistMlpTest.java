package io.leavesfly.tinyai.example.classify;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ndarr.NdArray;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 * MNIST模型测试类
 *
 * @author leavesfly
 * @version 0.01
 *
 * 加载训练好的模型，读取手写数字图片进行预测
 */
public class MnistMlpTest {

    public static void main(String[] args) {
        String imagePath = args.length > 0 ? args[0] : "test_digit.png";

        try {
            // 加载训练好的模型
            System.out.println("加载模型: models/mnist_mlp.model");
            Model model = Model.loadModel("models/mnist_mlp.model");
            System.out.println("模型加载成功！");

            // 读取并预处理图片
            System.out.println("读取图片: " + imagePath);
            NdArray input = preprocessImage(imagePath);

            // 进行预测
            System.out.println("开始预测...");
            Variable inputVar = new Variable(input);
            Variable outputVar = model.forward(inputVar);
            NdArray output = outputVar.getValue();

            // 获取预测结果
            int predictedDigit = getPredictedDigit(output);
            float confidence = getConfidence(output, predictedDigit);

            System.out.println("\n预测结果: " + predictedDigit);
            System.out.println("置信度: " + String.format("%.2f%%", confidence * 100));

            // 显示所有类别的概率
            System.out.println("\n各数字的概率分布:");
            printProbabilities(output);

        } catch (Exception e) {
            System.err.println("预测失败: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 预处理图片：读取图片并转换为28x28灰度图，归一化到[0,1]
     */
    private static NdArray preprocessImage(String imagePath) throws Exception {
        BufferedImage image = ImageIO.read(new File(imagePath));

        // 转换为灰度图并调整大小到28x28
        BufferedImage grayImage = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        grayImage.getGraphics().drawImage(image.getScaledInstance(28, 28, BufferedImage.SCALE_SMOOTH), 0, 0, null);

        // 转换为NdArray (28*28=784维向量)
        float[] pixels = new float[28 * 28];
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int rgb = grayImage.getRGB(x, y);
                int gray = (rgb >> 16) & 0xFF;
                // 归一化到[0,1]，MNIST是白底黑字，需要反转
                pixels[y * 28 + x] = (255 - gray) / 255.0f;
            }
        }

        return NdArray.of(pixels);
    }

    /**
     * 获取预测的数字（概率最大的类别）
     */
    private static int getPredictedDigit(NdArray output) {
        float[][] matrix = output.getMatrix();
        float maxProb = Float.NEGATIVE_INFINITY;
        int predictedDigit = -1;

        for (int i = 0; i < matrix[0].length; i++) {
            if (matrix[0][i] > maxProb) {
                maxProb = matrix[0][i];
                predictedDigit = i;
            }
        }

        return predictedDigit;
    }

    /**
     * 获取预测的置信度
     */
    private static float getConfidence(NdArray output, int predictedDigit) {
        float[][] matrix = output.getMatrix();
        return matrix[0][predictedDigit];
    }

    /**
     * 打印所有类别的概率分布
     */
    private static void printProbabilities(NdArray output) {
        float[][] matrix = output.getMatrix();
        for (int i = 0; i < matrix[0].length; i++) {
            System.out.println("  数字 " + i + ": " + String.format("%.2f%%", matrix[0][i] * 100));
        }
    }
}
