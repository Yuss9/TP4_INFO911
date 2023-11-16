#include <cstdio>
#include <iostream>
#include <algorithm>
#include <vector>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

struct ColorDistribution
{
    float data[8][8][8]; // l'histogramme
    int nb;              // le nombre d'échantillons

    ColorDistribution()
    {
        reset();
    }

    ColorDistribution &operator=(const ColorDistribution &other) = default;

    // Met à zéro l'histogramme
    void reset()
    {
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                for (int k = 0; k < 8; k++)
                {
                    data[i][j][k] = 0.0;
                }
            }
        }
        nb = 0;
    }

    // Ajoute l'échantillon color à l'histogramme:
    // met +1 dans la bonne case de l'histogramme et augmente le nb d'échantillons
    void add(Vec3b color)
    {
        // les rgb vont de 0 a 255 donc quand on divise par 32 sa ramene entre 0 et 7
        int i = color.val[0] / 32;
        int j = color.val[1] / 32;
        int k = color.val[2] / 32;
        data[i][j][k] += 1.0f;
        nb += 1;
    }

    // Indique qu'on a fini de mettre les échantillons:
    // divise chaque valeur du tableau par le nombre d'échantillons
    // pour que case représente la proportion des pixels qui ont cette couleur.
    void finished()
    {
        for (int i = 0; i < 8; ++i)
        {
            for (int j = 0; j < 8; ++j)
            {
                for (int k = 0; k < 8; ++k)
                {
                    data[i][j][k] /= static_cast<float>(nb);
                }
            }
        }
    }

    // Retourne la distance entre cet histogramme et l'histogramme other
    float distance(const ColorDistribution &other) const
    {
        float result = 0.0f;
        for (int i = 0; i < 8; ++i)
        {
            for (int j = 0; j < 8; ++j)
            {
                for (int k = 0; k < 8; ++k)
                {
                    float numerator = pow(data[i][j][k] - other.data[i][j][k], 2);
                    float denominator = data[i][j][k] + other.data[i][j][k];

                    // Éviter la division par zéro
                    if (denominator != 0.0f)
                    {
                        result += numerator / denominator;
                    }
                }
            }
        }
        return result;
    }
};

ColorDistribution getColorDistribution(Mat input, Point pt1, Point pt2)
{
    ColorDistribution cd;
    for (int y = pt1.y; y < pt2.y; y++)
    {
        for (int x = pt1.x; x < pt2.x; x++)
        {
            cd.add(input.at<Vec3b>(y, x));
        }
    }

    cd.finished();
    return cd;
}

float minDistance(const ColorDistribution &h, const std::vector<ColorDistribution> &hists)
{
    float minDist = std::numeric_limits<float>::max();

    for (const auto &hist : hists)
    {
        float dist = h.distance(hist);
        minDist = std::min(minDist, dist);
    }

    return minDist;
}

Mat recoObject(Mat input, const std::vector<ColorDistribution> &col_hists, const std::vector<ColorDistribution> &col_hists_object, const std::vector<Vec3b> &colors, const int bloc)
{
    Mat reco = Mat::zeros(input.rows, input.cols, CV_8UC3);

    for (int y = 0; y < input.rows; y += bloc)
    {
        for (int x = 0; x < input.cols; x += bloc)
        {
            Point pt1(x, y);
            Point pt2(x + bloc, y + bloc);

            ColorDistribution blockHist = getColorDistribution(input, pt1, pt2);

            float distBackground = minDistance(blockHist, col_hists);
            float distObject = minDistance(blockHist, col_hists_object);

            // Choix de la couleur en fonction de la distance minimale
            Vec3b color = (distBackground < distObject) ? colors[0] : colors[1];

            // Coloration du bloc dans l'image reco
            rectangle(reco, pt1, pt2, color, FILLED);
        }
    }

    return reco;
}

int main(int argc, char **argv)
{
    Mat img_input, img_seg, img_d_bgr, img_d_hsv, img_d_lab;
    VideoCapture *pCap = nullptr;
    const int width = 640;
    const int height = 480;
    const int size = 50;

    // Ouvre la camera
    pCap = new VideoCapture(0);
    if (!pCap->isOpened())
    {
        cout << "Couldn't open image / camera ";
        return 1;
    }

    // Force une camera 640x480 (pas trop grande).
    pCap->set(CAP_PROP_FRAME_WIDTH, 640);
    pCap->set(CAP_PROP_FRAME_HEIGHT, 480);
    (*pCap) >> img_input;
    if (img_input.empty())
        return 1; // probleme avec la camera

    Point pt1_left(0, 0);
    Point pt2_left(width / 2, height);
    Point pt1_right(width / 2, 0);
    Point pt2_right(width, height);

    namedWindow("input", 1);
    namedWindow("reco", 1);
    imshow("input", img_input);

    Point pt1(width / 2 - size / 2, height / 2 - size / 2);
    Point pt2(width / 2 + size / 2, height / 2 + size / 2);

    bool freeze = false;

    std::vector<Vec3b> colors;
    colors.push_back(Vec3b(0, 0, 0));   // Noir pour le fond
    colors.push_back(Vec3b(0, 0, 255)); // Rouge pour l'objet
    bool reco = false;                  // Indicateur du mode reconnaissance


    std::vector<ColorDistribution> col_hists;
    std::vector<ColorDistribution> col_hists_object;

    while (true)
    {
        char c = (char)waitKey(50); // attend 50ms -> 20 images/s

        if (pCap != nullptr && !freeze)
        {
            (*pCap) >> img_input; // récupère l'image de la caméra
        }

        if (c == 27 || c == 'q')
            break; // permet de quitter l'application

        if (c == 'f')
        { // permet de geler l'image
            freeze = !freeze;
        }

        if (c == 'v')
        {
            // 1) Calcule la distribution couleur de la partie gauche et de la partie droite de l'écran
            ColorDistribution cd_left = getColorDistribution(img_input, pt1_left, pt2_left);
            ColorDistribution cd_right = getColorDistribution(img_input, pt1_right, pt2_right);

            // 2) Calcule la distance entre les 2 distributions et l'affiche
            float distance = cd_left.distance(cd_right);
            cout << "La distance est : " << distance << endl;
        }


        if (c == 'b')
        {
            col_hists.clear();

            const int bbloc = 128;
            for (int y = 0; y <= height - bbloc; y += bbloc)
            {
                for (int x = 0; x <= width - bbloc; x += bbloc)
                {
                    Point pt1(x, y);
                    Point pt2(x + bbloc, y + bbloc);
                    ColorDistribution hist = getColorDistribution(img_input, pt1, pt2);
                    col_hists.push_back(hist);
                }
            }

            int nb_hists_background = col_hists.size();
            cout << "Nombre d'histogrammes de fond : " << nb_hists_background << endl;
        }

        if (c == 'a')
        {
            ColorDistribution obj_hist = getColorDistribution(img_input, pt1, pt2);
            col_hists_object.push_back(obj_hist);
            int nb_hists_object = col_hists_object.size();
            cout << "Nombre d'histogrammes d'objet : " << nb_hists_object << endl;
        }

        

        if (c == 'r')
        {
            reco = !reco;
            cout << "Mode reconnaissance : " << (reco ? "Activé" : "Désactivé") << endl;
        }

        Mat output = img_input;
        if (reco)
        {
            Mat gray;
            cvtColor(img_input, gray, COLOR_BGR2GRAY);
            Mat recoImg = recoObject(img_input, col_hists, col_hists_object, colors, 8);
            cvtColor(gray, img_input, COLOR_GRAY2BGR);
            output = 0.5 * recoImg + 0.5 * img_input; 
        }
        else
        {
            cv::rectangle(img_input, pt1, pt2, Scalar({255.0, 255.0, 255.0}), 1);
        }

        imshow( "reco", output );


        cv::rectangle(img_input, pt1_left, pt2_left, Scalar({255.0, 255.0, 255.0}), 1);
        cv::rectangle(img_input, pt1_right, pt2_right, Scalar({255.0, 255.0, 255.0}), 1);
        cv::rectangle(img_input, pt1, pt2, Scalar({255.0, 255.0, 255.0}), 1);
        imshow("input", img_input); // affiche le flux video
    }

    return 0;
}