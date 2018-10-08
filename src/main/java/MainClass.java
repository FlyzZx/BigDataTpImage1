import org.bytedeco.javacpp.*;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;

import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.NORM_L2;
import static org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_features2d.drawKeypoints;
import static org.bytedeco.javacpp.opencv_features2d.drawMatches;
import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.CV_COMP_BHATTACHARYYA;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class MainClass {

    public static void main(String[] args) {
        Mat image = imread("data/tower.jpg", IMREAD_COLOR);
        resize(image, image, new Size(800, 600));

        JFrame frame = new MainForm();
        frame.setSize(new Dimension(800, 600));
        frame.setVisible(true);
    }

    private static void siftMatching(Mat image) {
        Mat secondImage = imread("data/tower.jpg");

        opencv_core.KeyPointVector keyPoints = new opencv_core.KeyPointVector();
        opencv_core.KeyPointVector keyPoints2 = new opencv_core.KeyPointVector();
        int nFeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.03;
        int edgeThreshold = 10;
        double sigma = 1.6;
        Loader.load(opencv_calib3d.class);
        Loader.load(opencv_shape.class);

        opencv_xfeatures2d.SIFT sift;

        sift = opencv_xfeatures2d.SIFT.create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        //First Image
        sift.detect(image, keyPoints);
        Mat descriptor = new Mat();
        sift.compute(image, keyPoints, descriptor);

        //Train Images
        sift.detect(secondImage, keyPoints2);
        Mat descriptorSecondImage = new Mat();
        sift.compute(secondImage, keyPoints2, descriptorSecondImage);


        Mat featureImage = new Mat();
        Mat featureImage2 = new Mat();
        drawKeypoints(image, keyPoints, featureImage, new opencv_core.Scalar(255, 255, 255, 0),
                opencv_features2d.DrawMatchesFlags.DRAW_RICH_KEYPOINTS);
        drawKeypoints(secondImage, keyPoints2, featureImage2, new opencv_core.Scalar(255, 255, 255, 0),
                opencv_features2d.DrawMatchesFlags.DRAW_RICH_KEYPOINTS);

        Show(featureImage, "Image with SIFT");
        Show(featureImage2, "Image with SIFT");

        opencv_features2d.BFMatcher matcher = new opencv_features2d.BFMatcher(NORM_L2, false);
        opencv_core.DMatchVector matches = new opencv_core.DMatchVector();
        matcher.match(descriptor, descriptorSecondImage, matches);

        opencv_core.DMatchVector bestMatches = selectBest(matches, 100);

        Mat imageMatches = new Mat();
        byte[] mask = null;
        drawMatches(image, keyPoints, secondImage, keyPoints2,
                bestMatches, imageMatches, new opencv_core.Scalar(0, 0, 255, 0), new opencv_core.Scalar(255, 0, 0, 0), mask,
                opencv_features2d.DrawMatchesFlags.DEFAULT);

        Show(imageMatches, "Matches");
    }

    private static opencv_core.DMatchVector selectBest(opencv_core.DMatchVector matches, int numberToSelect) {
        opencv_core.DMatch[] sorted = toArray(matches);
        Arrays.sort(sorted, (a, b) -> {
            return a.lessThan(b) ? -1 : 1;
        });
        opencv_core.DMatch[] best = Arrays.copyOf(sorted, numberToSelect);
        return new opencv_core.DMatchVector(best);
    }

    private static opencv_core.DMatch[] toArray(opencv_core.DMatchVector matches) {
        assert matches.size() <= Integer.MAX_VALUE;
        int n = (int) matches.size();
        // Convert keyPoints to Scala sequence
        opencv_core.DMatch[] result = new opencv_core.DMatch[n];
        for (int i = 0; i < n; i++) {
            result[i] = new opencv_core.DMatch(matches.get(i));
        }
        return result;
    }

    private static void compareImage(opencv_core.Mat imgTarget, String pathRef) {
        ArrayList<Mat> descriptorsRef = new ArrayList<opencv_core.Mat>();
        File trainFolder = new File(pathRef);
        if (trainFolder.isDirectory()) {
            System.out.println("Starting train...");
            for (String file : trainFolder.list()) {
                opencv_core.Mat ref = imread(pathRef + file, IMREAD_GRAYSCALE);
                System.out.println("Opening " + file);
                resize(ref, ref, new opencv_core.Size(800, 600));
                //histogramme(ref);
                descriptorsRef.add(getHistogramToMat(ref));
            }

            System.out.println("Predicting...");
            opencv_core.Mat histoTarget = getHistogramToMat(imgTarget);
            double[] prediction = new double[descriptorsRef.size()];
            for (int i = 0; i < descriptorsRef.size(); i++) {
                opencv_core.Mat mat = descriptorsRef.get(i);
                System.out.println("Prediction for " + i + " : " + Math.round((1 - compareHistogram(mat, histoTarget, CV_COMP_BHATTACHARYYA)) * 100) + "%");
            }
        }
    }

}
