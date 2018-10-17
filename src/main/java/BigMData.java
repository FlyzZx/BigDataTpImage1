import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.TreeMap;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_features2d.drawKeypoints;
import static org.bytedeco.javacpp.opencv_features2d.drawMatches;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class BigMData {

    public static void Show(opencv_core.Mat mat, String title) {
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        CanvasFrame canvas = new CanvasFrame(title, 1);
        canvas.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        canvas.showImage(converter.convert(mat));
    }

    public static void showMat(opencv_core.Mat m) {

        UByteRawIndexer ind = m.createIndexer();
        for (int rows = 0; rows < m.rows(); rows++) {
            for (int cols = 0; cols < m.cols(); cols++) {
                System.out.print(ind.get(rows, cols) + " ");
            }
            System.out.println("");
        }
    }

    public static void segmentationKmeans(opencv_core.Mat image) {
        //Reshape en vecteur n-dims
        opencv_core.Mat reshaped_image = image.reshape(1, image.cols() * image.rows());
        opencv_core.Mat reshaped_image32f = new opencv_core.Mat();
        reshaped_image.convertTo(reshaped_image32f, CV_32F);

        //KMEANS.
        int clusterCount = 4;
        opencv_core.Mat labels = new opencv_core.Mat();
        opencv_core.Mat centers = new opencv_core.Mat();
        opencv_core.TermCriteria criteria = new opencv_core.TermCriteria(opencv_core.TermCriteria.MAX_ITER + opencv_core.TermCriteria.EPS, 10, 1.0);
        kmeans(reshaped_image32f, clusterCount, labels, criteria, 10, KMEANS_PP_CENTERS, centers);

        //Reshape des labels pour affichage
        centers.convertTo(centers, CV_8U);
        Show(centers, "Clustering centers");

        Mat labelsShow = new Mat();
        labels.convertTo(labelsShow, CV_8U, 256 / clusterCount, 3);
        labels.convertTo(labels, CV_8U);

        UByteRawIndexer centerIndexer = centers.createIndexer();

        labels = labels.reshape(1, image.rows());
        labelsShow = labelsShow.reshape(1, image.rows());
        Show(labelsShow, "labels");
        UByteRawIndexer labelsIndexer = labels.createIndexer();

        int centerHeight = (int) centerIndexer.sizes()[0];
        int centerWidth = (int) centerIndexer.sizes()[1];
        ArrayList<int[]> colors = new ArrayList<int[]>();
        for (int i = 0; i < centerHeight; i++) {
            int[] temp = new int[centerWidth];
            temp[0] = centerIndexer.get(i, 0);
            temp[1] = centerIndexer.get(i, 1);
            temp[2] = centerIndexer.get(i, 2);
            colors.add(temp);
        }

        Mat r = new Mat(600, 800, CV_8UC1);
        UByteRawIndexer rIndexer = r.createIndexer();
        Mat g = new Mat(600, 800, CV_8UC1);
        UByteRawIndexer gIndexer = g.createIndexer();
        Mat b = new Mat(600, 800, CV_8UC1);
        UByteRawIndexer bIndexer = b.createIndexer();

        for (int i = 0; i < labels.rows(); i++) {
            for (int j = 0; j < labels.cols(); j++) {
                int[] couleur = colors.get(labelsIndexer.get(i, j));
                rIndexer.put(i, j, couleur[0]);
                gIndexer.put(i, j, couleur[1]);
                bIndexer.put(i, j, couleur[2]);
            }
        }

        Show(r, "r");
        Show(g, "g");
        Show(b, "b");

        Mat resultats = new Mat(600, 800, CV_8UC3);

        MatVector rgb = new MatVector();
        rgb.push_back(r);
        rgb.push_back(g);
        rgb.push_back(b);
        merge(rgb, resultats);
        UByteRawIndexer resultIndexerDebug = resultats.createIndexer();
        Show(resultats, "Quantization results");
    }

    public static void withLut(opencv_core.Mat mat) {
        System.out.println("With inversed lut");
        opencv_core.Mat inversedLutImage = mat.clone();
        LUT(mat, new opencv_core.Mat(buildInversedLut()), inversedLutImage);
        inversedLutImage.convertTo(inversedLutImage, CV_8U);
        Show(inversedLutImage, "With inversed LUT");

        System.out.println("With custom lut");
        opencv_core.Mat customLutImage = mat.clone();
        LUT(mat, new opencv_core.Mat(buildCustomLut()), customLutImage);
        customLutImage.convertTo(customLutImage, CV_8U);
        Show(customLutImage, "With custom LUT");

        System.out.println("With custom 2 lut");
        opencv_core.Mat customTwoLutImage = mat.clone();
        LUT(mat, new opencv_core.Mat(buildCustomTwoLut()), customTwoLutImage);
        customTwoLutImage.convertTo(customTwoLutImage, CV_8U);
        Show(customTwoLutImage, "With custom 2 LUT");

        System.out.println("With custom function lut");
        //opencv_core.Mat lutWithCustomFunction = mat.clone();
        //LUT(mat, new opencv_core.Mat(buildCustomTwoLut()), lutWithCustomFunction);
        Mat afterCustomLut = applyLut(mat, buildCustomTwoLut());
        //afterCustomLut.convertTo(afterCustomLut, CV_8U);
        Show(afterCustomLut, "With custom function LUT");
    }

    public static void matchTemplateTest() {
        opencv_core.Mat image1 = imread("data/Mannekenpis/mannekenpis2.jpeg");
        resize(image1, image1, new opencv_core.Size(800, 600));
        opencv_core.Mat image2 = imread("data/Mannekenpis/mannekenpis1.jpeg");
        resize(image2, image2, new opencv_core.Size(800, 600));
        opencv_core.Mat target = new opencv_core.Mat(image1, new opencv_core.Rect(385, 130, 70, 60));
        Show(target, "Template");
        rectangle(image1, new opencv_core.Rect(385, 130, 70, 60), opencv_core.Scalar.YELLOW);
        Show(image1, "Original");
        opencv_core.Mat result = new opencv_core.Mat();
        matchTemplate(
                image2, // search region
                target, // template
                result, // result
                CV_COMP_BHATTACHARYYA);
        double[] minVal = new double[1];
        double[] maxVal = new double[1];
        opencv_core.Point minPt = new opencv_core.Point();
        opencv_core.Point maxPt = new opencv_core.Point();
        minMaxLoc(result, minVal, maxVal, minPt, maxPt, new Mat());
        System.out.println("minPt = (" + minPt.x() + ", " + minPt.y() + ")");
        rectangle(image2, new opencv_core.Rect(maxPt.x(), maxPt.y(), target.cols(), target.rows()), opencv_core.Scalar.CYAN);
        Show(image2, "Best match");
        Show(result, "result");
    }

    public static void histogrammeInteret(opencv_core.Mat image) {
        //Découpe les rectangles sur l'images
        opencv_core.Mat target1 = new opencv_core.Mat(image, new opencv_core.Rect(120, 40, 30, 30));
        opencv_core.Mat target2 = new opencv_core.Mat(image, new opencv_core.Rect(250, 80, 30, 30));
        opencv_core.Mat target3 = new opencv_core.Mat(image, new opencv_core.Rect(500, 420, 50, 50));
        //Crée un histogramme pour chaque rectangle
        opencv_core.Mat targetHistoMat1 = getHistogramToMat(target1);
        opencv_core.Mat targetHistoMat2 = getHistogramToMat(target2);
        opencv_core.Mat targetHistoMat3 = getHistogramToMat(target3);
        //Dessine les rectangles sur l'image
        rectangle(image, new opencv_core.Rect(120, 40, 30, 30), opencv_core.Scalar.YELLOW);
        rectangle(image, new opencv_core.Rect(250, 80, 30, 30), opencv_core.Scalar.GREEN);
        rectangle(image, new opencv_core.Rect(500, 420, 50, 50), opencv_core.Scalar.BLUE);

        System.out.println("Similarité entre jaune et vert :" + Math.round((1 - compareHistogram(targetHistoMat1, targetHistoMat2, CV_COMP_BHATTACHARYYA)) * 100) + " %");
        System.out.println("Similarité entre jaune et bleu :" + Math.round((1 - compareHistogram(targetHistoMat1, targetHistoMat3, CV_COMP_BHATTACHARYYA)) * 100) + " %");
        System.out.println("Similarité entre vert et bleu :" + Math.round((1 - compareHistogram(targetHistoMat2, targetHistoMat3, CV_COMP_BHATTACHARYYA)) * 100) + " %");


        Show(image, "Original");
        showHistogram(getHistogram(target1), "Rectangle Jaune", Color.yellow);
        showHistogram(getHistogram(target2), "Rectangle Vert", Color.green);
        showHistogram(getHistogram(target3), "Rectangle Bleu", Color.blue);
    }

    public static Mat applyLut(Mat image, float[] lut) {
        Mat resultMat = image.clone();
        UByteIndexer indexer = resultMat.createIndexer();
        for (int x = 0; x < indexer.height(); x++) {
            for (int y = 0; y < indexer.width(); y++) {
                indexer.put(x, y, (int) lut[indexer.get(x, y)]);
            }
        }
        return resultMat;
    }

    public static float[] buildInversedLut() {
        float[] lut = new float[256];
        for (int i = 0; i < lut.length; i++) {
            lut[i] = 255 - i;
        }
        return lut;
    }

    public static float[] buildCustomLut() {
        float[] lut = new float[256];
        for (int i = 0; i < lut.length; i++) {
            if (i <= 17) {
                lut[i] = 0;
            } else if (i >= 102) {
                lut[i] = 255;
            } else {
                lut[i] = 3 * i;
            }
        }
        return lut;
    }

    public static float[] buildCustomTwoLut() {
        float[] lut = new float[256];
        for (int i = 0; i < lut.length; i++) {
            //y=(cos(((float)x/255)*2*PI - PI)+1)*255
            lut[i] = (float) ((Math.cos(((float) i / 255) * 2 * Math.PI - Math.PI) + 1) * 255);
            if (lut[i] > 255) {
                lut[i] = 255;
            }
        }
        return lut;
    }

    public static void compareImage(opencv_core.Mat imgTarget, String pathRef) {
        ArrayList<opencv_core.Mat> descriptorsRef = new ArrayList<opencv_core.Mat>();
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

    public static void siftTraining(Mat image) {
        //Calcul Sift pour l'image de test
        System.out.println("Extracting descriptors from test picture...");
        TreeMap<String, Mat> trainDescriptors = new TreeMap<>();
        KeyPointVector kpTest = new KeyPointVector();
        Mat descriptorsTest = new Mat();
        int nFeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.03;
        int edgeThreshold = 10;
        double sigma = 1.6;
        Loader.load(opencv_calib3d.class);
        Loader.load(opencv_shape.class);

        opencv_xfeatures2d.SIFT sift;

        sift = opencv_xfeatures2d.SIFT.create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        sift.detect(image, kpTest);
        sift.compute(image, kpTest, descriptorsTest);

        //Récupération des descripteurs de référence
        System.out.println("Extracting descriptors from train pictures...");
        File srcData = new File("data/Train/");
        DMatchVector matches = new DMatchVector();
        opencv_features2d.BFMatcher matcher = new opencv_features2d.BFMatcher(NORM_L2, false);
        for (File folder : srcData.listFiles()) {
            for (File f : folder.listFiles()) {
                //System.out.println("Extracting for " + f.getName() + "...");
                Mat current = imread(f.getAbsolutePath(), IMREAD_COLOR);
                resize(current, current, new opencv_core.Size(800, 600));
                KeyPointVector kpCurrent = new KeyPointVector();
                Mat descCurrent = new Mat();

                sift.detect(current, kpCurrent);
                sift.compute(current, kpCurrent, descCurrent);
                //trainDescriptors.put(f.getName(), descCurrent);
                DMatchVector m = new DMatchVector();
                matcher.match(descriptorsTest, descCurrent, m);
                m = selectBest(m, 25);

                double moyenne = 0.0;
                for (DMatch match : m.get()) {
                    moyenne += match.distance();
                }

                moyenne = moyenne / m.size();
                //matches.put(m);
                //System.out.println("Matches found for " + f.getName() + " : " + m.size());
                //Calcul distance euclidienne
                System.out.println("Distance for " + f.getName() + " : " + moyenne);
            }
        }
    }

    public static void histogramme(opencv_core.Mat image) {
        opencv_core.Mat gray = new opencv_core.Mat(image.size());
        cvtColor(image, gray, CV_RGB2GRAY);
        Show(gray, "Gray");

        showHistogram(getHistogram(image), "Histogramme", Color.blue);
    }

    public static void showHistogram(float[] hist, String caption, Color couleur) {
        int numberOfBins = 256;
        //	Output	image	size
        int width = numberOfBins;
        int height = numberOfBins;
        //	Set	highest	point	to	90%	of	the	number	of	bins
        double scale = 0.9 / max(hist) * height;
        //	Create	a	color	image	to	draw	on
        BufferedImage canvas = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = canvas.createGraphics();
        //	Paint	background
        g.setPaint(Color.WHITE);
        g.fillRect(0, 0, width, height);
        //	Draw	a	vertical	line	for	each	bin
        g.setPaint(couleur);
        for (int bin = 0; bin < numberOfBins; bin++) {
            int h = (int) Math.round(hist[bin] * scale);
            g.drawLine(bin, height - 1, bin, height - h - 1);
        }
        //	Cleanup
        g.dispose();
        //	Create	an	image	and	show	the	histogram
        CanvasFrame canvasF = new CanvasFrame(caption, 1);
        canvasF.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        canvasF.showImage(canvas);
    }

    public static float max(float[] tab) {
        float max = Float.MIN_VALUE;
        for (int i = 0; i < tab.length; i++) {
            if (max < tab[i]) {
                max = tab[i];
            }
        }
        return max;
    }

    public static float[] getHistogram(opencv_core.Mat image) {
        float[] histo = new float[256];

        for (int i = 0; i < histo.length; i++) //INIT RESULT ARRAY TO ZERO
            histo[i] = 0;

        UByteIndexer indexer = (UByteIndexer) image.createIndexer();
        for (int x = 0; x < indexer.width(); x++) {
            for (int y = 0; y < indexer.height(); y++) {
                int value = indexer.get(y, x);
                if (value < 256) histo[value]++;
            }
        }
        return histo;
    }

    public static opencv_core.Mat getHistogramToMat(opencv_core.Mat image) {
        float[] histo = getHistogram(image);
        opencv_core.Mat m = new opencv_core.Mat(histo);
        return m;
    }

    public static double compareHistogram(opencv_core.Mat histo1, opencv_core.Mat histo2, int comparisonMethod) {
        double similarite = compareHist(histo1, histo2, comparisonMethod);
        //System.out.println("Similarité :" + similarite);
        return similarite;
    }

    public static void wreckedtomestleseulRGB(opencv_core.Mat image) {
        opencv_core.MatVector rgbSplit = new opencv_core.MatVector();
        split(image, rgbSplit);
        Show(rgbSplit.get(0), "Red");
        showHistogram(getHistogram(rgbSplit.get(0)), "Histo Red", Color.RED);
        Show(rgbSplit.get(1), "Green");
        showHistogram(getHistogram(rgbSplit.get(1)), "Histo Green", Color.GREEN);
        Show(rgbSplit.get(2), "Blue");
        showHistogram(getHistogram(rgbSplit.get(2)), "Histo Blue", Color.BLUE);


    }

    public static void morpho(opencv_core.Mat image) {
        //Thresh
        opencv_core.Mat thresh = new opencv_core.Mat(image.size());
        threshold(image, thresh, 120, 255, THRESH_BINARY_INV);
        Show(thresh, "Thresh");

        //Erode
        opencv_core.Mat element = new opencv_core.Mat(5, 5, CV_8U, new opencv_core.Scalar(1d));
        opencv_core.Mat erosion = new opencv_core.Mat(image.size());
        erode(thresh, erosion, element);
        Show(erosion, "Erode");

        //MorphologyEx
        opencv_core.Mat opened = new opencv_core.Mat(image.size());
        morphologyEx(thresh, opened, MORPH_CLOSE, element);
        Show(opened, "Closed");
    }

    public static void siftMatching(Mat image) {
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

    public static opencv_core.DMatchVector selectBest(opencv_core.DMatchVector matches, int numberToSelect) {
        opencv_core.DMatch[] sorted = toArray(matches);
        Arrays.sort(sorted, (a, b) -> {
            return a.lessThan(b) ? -1 : 1;
        });
        opencv_core.DMatch[] best = Arrays.copyOf(sorted, numberToSelect);
        return new opencv_core.DMatchVector(best);
    }

    public static opencv_core.DMatch[] toArray(opencv_core.DMatchVector matches) {
        assert matches.size() <= Integer.MAX_VALUE;
        int n = (int) matches.size();
        // Convert keyPoints to Scala sequence
        opencv_core.DMatch[] result = new opencv_core.DMatch[n];
        for (int i = 0; i < n; i++) {
            result[i] = new opencv_core.DMatch(matches.get(i));
        }
        return result;
    }

    public static void splitRGBShow(opencv_core.Mat image, boolean R, boolean G, boolean B) {
        opencv_core.MatVector rgbSplit = new opencv_core.MatVector();
        opencv_core.MatVector choosenSplit = new opencv_core.MatVector();
        opencv_core.Mat red = new opencv_core.Mat(image.rows(), image.cols(), CV_8UC1);
        opencv_core.Mat green = new opencv_core.Mat(image.rows(), image.cols(), CV_8UC1);
        opencv_core.Mat blue = new opencv_core.Mat(image.rows(), image.cols(), CV_8UC1);
        opencv_core.Mat result = new opencv_core.Mat();
        String windowName = "";
        split(image, rgbSplit);

        if (R) {
            red = rgbSplit.get(2);
            windowName += "R";
        }
        if (G) {
            green = rgbSplit.get(1);
            windowName += "G";
        }
        if (B) {
            blue = rgbSplit.get(0);
            windowName += "B";
        }

        choosenSplit.push_back(blue);
        choosenSplit.push_back(green);
        choosenSplit.push_back(red);
        merge(choosenSplit, result);
        String name = windowName.replaceAll("(?<=.)(?=.)", "+");
        Show(result, name);
    }

    public static void detectFace(opencv_core.Mat image) {
        opencv_objdetect.CascadeClassifier face_cascade = new opencv_objdetect.CascadeClassifier("data/resources/haarcascade_frontalface_alt.xml");
        opencv_objdetect.CascadeClassifier smile_cascade = new opencv_objdetect.CascadeClassifier("data/resources/haarcascades/haarcascade_eye.xml");

        RectVector faces = new RectVector();
        RectVector smiles = new RectVector();
        // Find the faces in “image”:
        face_cascade.detectMultiScale(image, faces);
        smile_cascade.detectMultiScale(image, smiles);

        for (int i = 0; i < faces.size(); i++) {
            Rect face_i = faces.get(i);
            Mat face = new Mat(image, face_i);
            rectangle(image, face_i, new Scalar(0, 255, 0, 1));
        }
        for (int i = 0; i < smiles.size(); i++) {
            Rect smile = smiles.get(i);
            Mat face = new Mat(image, smile);
            rectangle(image, smile, new Scalar(255, 0, 0, 1));
        }
        // Show the result:
        Show(image, "Détection des visages");
    }

    public static void trainFace(Mat image) {
        System.out.println("Training...");
        opencv_face.FaceRecognizer faceRecognizer = opencv_face.LBPHFaceRecognizer.create();
        String trainingDir = "data/Face/Dection_faces";
        File root = new File(trainingDir);
        FilenameFilter imgFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
            }
        };

        //File[] imageFiles = root.listFiles(imgFilter);


        MatVector images = new MatVector(root.listFiles().length);

        Mat labels = new Mat(root.listFiles().length, 1, CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();

        int counter = 0;

        HashMap<String, Integer> tamere = new HashMap<>();
        tamere.put("benj;1", 1);
        tamere.put("jor", 2);
        tamere.put("thom", 3);
        tamere.put("andre", 4);
        tamere.put("dam", 5);


        for (File im : root.listFiles()) {
            //System.out.println("path:" + image.getAbsolutePath());
            Mat img = imread(im.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            //System.out.println(Integer.parseInt(im.getName().split("-.")[0]));
            int label = Integer.parseInt(im.getName().split("_")[0]);
            images.put(counter, img);
            labelsBuf.put(counter, label);
            counter++;
            //  System.out.println("path:" + counter);
        }

        faceRecognizer.train(images, labels);
        faceRecognizer.save("data/training.xml");

        System.out.println("Testing...");
        faceRecognizer = opencv_face.LBPHFaceRecognizer.create();
        faceRecognizer.read("data/training.xml");

        int prediction = faceRecognizer.predict_label(image);
        System.out.println("prediction:" + prediction);
    }
}
