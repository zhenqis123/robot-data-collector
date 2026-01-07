#include <iostream>

#include <librealsense2/rs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    try
    {
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, RS2_FORMAT_BGR8);
        cfg.enable_stream(RS2_STREAM_DEPTH, RS2_FORMAT_Z16);
        rs2::align align_to_color(RS2_STREAM_COLOR);

        auto profile = pipe.start(cfg);
        auto dev = profile.get_device();
        std::cout << "RealSense device: "
                  << dev.get_info(RS2_CAMERA_INFO_NAME) << " ("
                  << dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << ")\n";

        cv::namedWindow("RealSense RGB", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("RealSense Depth", cv::WINDOW_AUTOSIZE);

        while (true)
        {
            rs2::frameset frames = pipe.wait_for_frames();
            frames = align_to_color.process(frames);
            auto color = frames.get_color_frame();
            auto depth = frames.get_depth_frame();

            cv::Mat colorMat(cv::Size(color.get_width(), color.get_height()), CV_8UC3,
                             const_cast<void *>(color.get_data()), cv::Mat::AUTO_STEP);
            cv::Mat depthMat(cv::Size(depth.get_width(), depth.get_height()), CV_16U,
                             const_cast<void *>(depth.get_data()), cv::Mat::AUTO_STEP);

            cv::Mat colorImage = colorMat.clone();
            cv::Mat depthDisplay;
            depthMat.convertTo(depthDisplay, CV_8U, 255.0 / 8000.0);
            cv::applyColorMap(depthDisplay, depthDisplay, cv::COLORMAP_JET);

            const auto distance = depth.get_distance(depth.get_width() / 2, depth.get_height() / 2);
            cv::putText(colorImage,
                        "Center distance: " + std::to_string(distance).substr(0, 5) + "m",
                        {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);

            cv::imshow("RealSense RGB", colorImage);
            cv::imshow("RealSense Depth", depthDisplay);

            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q')
                break;
        }

        cv::destroyAllWindows();
        return 0;
    }
    catch (const rs2::error &e)
    {
        std::cerr << "RealSense error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
