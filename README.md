# aesthetic_image_pytorch

aesthetic_imageのPytorch実装(開発中)

EfficientNet-B0によるAVAデータセットの美しさのスコア付け

<details>
<summary>学習用コード等</summary>
 
|ファイル名|説明|
|----|----|
|convert_annotation.py|アノテーションファイルを変換するコード．|
|data_loader.py|データセットクラスとデータローダークラスのコード．|
|make_graph.py|学習曲線を可視化するコード．|
|main.py|学習，評価を開始するコード．|
|engine.py|学習・検証・評価のループのコード．|
</details>

## 実行手順

### AVAデータセットのダウンロード

下記からAVAデータセットをダウンロードする．

https://github.com/mtobeiyf/ava_downloader.


### アノテーションファイルの変換
```
python3 convert_annotation.py --annotation_path /path/to/AVA_dataset/AVA.txt --image_dir /path/to/images
```

### 学習
ハイパーパラメータは適宜調整してください．

```
python3 main.py --epoch 30 --batch_size 32 --img_dir ./images --train_csv ./AVA_train.csv --validation_csv ./AVA_validation.csv --amp --phase train
```

### 評価
`model_name.tar`には学習済みのモデルのパスを指定してください．

```
python3 main.py --batch_size 32 --img_dir ./images --validation_csv ./AVA_validation.csv --amp --phase test --test_model model_name.tar
```

## 参考文献
* Qiita「TPUを使った写真の美しさスコア付けモデルの学習」
    * https://qiita.com/kasagon/items/2fcada1d3b6c266bff5e#%E6%A6%82%E8%A6%81
    * https://github.com/myzkyuki/aesthetic_image/tree/master
