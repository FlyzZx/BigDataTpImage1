import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;

import static jdk.nashorn.internal.objects.NativeMath.max;
import static org.bytedeco.javacpp.opencv_core.CV_8U;
import static org.bytedeco.javacpp.opencv_core.split;
import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class MainClass {

    public static void Show(opencv_core.Mat mat, String title) {
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        CanvasFrame canvas = new CanvasFrame(title, 1);
        canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        canvas.showImage(converter.convert(mat));

    }

    private void showMat(opencv_core.Mat m) {

        FloatRawIndexer ind = m.createIndexer();
        for (int rows = 0; rows < m.rows(); rows++) {
            for (int cols = 0; cols < m.cols(); cols++) {
                System.out.print(ind.get(rows, cols));
            }
            System.out.println("");
        }
    }

    public static void main(String[] args) {
        opencv_core.Mat image = imread("data/tower.jpg", IMREAD_GRAYSCALE);
        resize(image,image,new opencv_core.Size(800,600));
        if (image == null || image.empty()) {
            return;
        }

        //System.out.println("image" + image.cols() + "	x	" + image.rows());
        //Show(image, "original");

        //morpho(image);
        //wreckedtomestleseulRGB(image);
        //histogramme(image);
        compareImage(image, "data/Monkey/");

    }

    private static void compareImage(opencv_core.Mat imgTarget, String pathRef) {
        ArrayList<opencv_core.Mat> descriptorsRef = new ArrayList<opencv_core.Mat>();
        File trainFolder = new File(pathRef);
        if(trainFolder.isDirectory()) {
            System.out.println("Starting train...");
            for (String file : trainFolder.list()) {
                opencv_core.Mat ref = imread(pathRef + file, IMREAD_GRAYSCALE);
                System.out.println("Opening " + file);
                resize(ref,ref,new opencv_core.Size(800,600));
                //histogramme(ref);
                descriptorsRef.add(getHistogramToMat(ref));
            }

            System.out.println("Predicting...");
            opencv_core.Mat histoTarget = getHistogramToMat(imgTarget);
            double[] prediction = new double[descriptorsRef.size()];
            for(int i = 0; i < descriptorsRef.size(); i++) {
                opencv_core.Mat mat = descriptorsRef.get(i);
                System.out.println("Prediction for " + i + " : " + Math.round((1 - compareHistogram(mat, histoTarget, CV_COMP_BHATTACHARYYA  )) * 100) + "%");
            }
        }
    }

    private static void histogramme(opencv_core.Mat image) {
        opencv_core.Mat gray = new opencv_core.Mat(image.size());
        cvtColor(image, gray, CV_RGB2GRAY);
        Show(gray, "Gray");

        showHistogram(getHistogram(image), "Histogramme",Color.blue);
    }

    public static void showHistogram(float[] hist, String caption,Color couleur ) {
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

    public static float max(float[] tab){
       float max = Float.MIN_VALUE;
        for(int i=0 ; i<tab.length;i++){
            if(max < tab[i]){
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
                if(value < 256) histo[value]++;
            }
        }
        return histo;
    }
    public static opencv_core.Mat getHistogramToMat(opencv_core.Mat image){
        float[] histo = getHistogram(image);
        opencv_core.Mat m = new opencv_core.Mat(histo);
        return m;
    }

    private  static double compareHistogram(opencv_core.Mat histo1, opencv_core.Mat histo2, int comparisonMethod){
        double similarite = compareHist(histo1,histo2,comparisonMethod);
        //System.out.println("Similarité :" + similarite);
        return similarite;
    }

    private static void wreckedtomestleseulRGB(opencv_core.Mat image) {
        opencv_core.MatVector rgbSplit = new opencv_core.MatVector();
        split(image, rgbSplit);
        Show(rgbSplit.get(0), "Red");
        showHistogram(getHistogram(rgbSplit.get(0)),"Histo Red",Color.RED);
        Show(rgbSplit.get(1), "Green");
        showHistogram(getHistogram(rgbSplit.get(1)),"Histo Green",Color.GREEN);
        Show(rgbSplit.get(2), "Blue");
        showHistogram(getHistogram(rgbSplit.get(2)),"Histo Blue",Color.BLUE);


    }

    private static void morpho(opencv_core.Mat image) {
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
}
