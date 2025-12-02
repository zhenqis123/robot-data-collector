#include <QCoreApplication>
#include <QMediaPlayer>
#include <QUrl>

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);

    QString file = "/home/xiziheng/develop/data_collector/piper_output.wav";
    if (argc > 1)
        file = argv[1];

    auto *player = new QMediaPlayer(&app);
    player->setMedia(QUrl::fromLocalFile(file));
    player->setVolume(100);
    player->play();

    QObject::connect(player, &QMediaPlayer::stateChanged, &app, [&](QMediaPlayer::State state) {
        if (state == QMediaPlayer::StoppedState)
            app.quit();
    });

    return app.exec();
}
