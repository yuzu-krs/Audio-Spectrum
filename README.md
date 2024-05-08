# Audio-Spectrum

- これは、Python でオーディオスペクトラム表示するプログラムです。
- 以下の input_device_index の値を変えることで仮想マイクやマイクをつなげることができます。

```
  audio = pyaudio.PyAudio()
  stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
  input=True, input_device_index=2, frames_per_buffer=FRAME_SIZE)
```

- 以下のプログラムを編集することで、棒状と円状のスペクトラムを設定できます。
- 初期プログラムは、円状に設定しております。コメントアウトして調節してください。

```
# 出力処理
cv2.rectangle(img, (0,0), (WIDTH, HEIGHT), (0,255,0), thickness=-1)   # 背景を緑に
 for index, value in enumerate(spectram_data):
    # 白色の波形として表示
     cv2.rectangle(img,
                 (int(part_w * (index + 0) + 1), int(HEIGHT)),
                 (int(part_w * (index + 1) - 1), int(max(HEIGHT - value/4, 0))),
                (255, 255, 255), thickness=-1)  # 波形を白に
画面表示
 cv2.imshow("Microphone Test", img)
```

```
        # 出力処理
    cv2.rectangle(img, (0,0), (WIDTH, HEIGHT), (0,255,0), thickness=-1)   # 出力領域のクリア
    for index, value in enumerate(spectram_data):
        # 単色のグラフとして表示
        rad = (2 * np.pi) * (index / len(spectram_data))
        x1 = int(WIDTH / 2 + np.sin(rad) * 80)
        y1 = int(HEIGHT / 2 - np.cos(rad) * 80)
        rad = (2 * np.pi) * (index / len(spectram_data))
        x2 = int(WIDTH / 2 + np.sin(rad) * (80 + value/4))
        y2 = int(HEIGHT / 2 - np.cos(rad) * (80 + value/4))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
    # 画面表示
    cv2.imshow("Microphone Test", img)
```
