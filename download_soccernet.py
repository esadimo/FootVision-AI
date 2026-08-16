from SoccerNet.Downloader import SoccerNetDownloader

downloader = SoccerNetDownloader(
    LocalDirectory="data/SoccerNet"
)

downloader.downloadDataTask(
    task="tracking",
    split=["train"]
)