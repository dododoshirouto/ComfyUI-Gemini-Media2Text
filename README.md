# ComfyUI-Gemini-Media2Text

自分の Gemini API キーで、画像・動画・フレーム列をテキスト化する ComfyUI カスタムノード。

## インストール（Windows コマンドプロンプト）
```
cd ComfyUI\custom_nodes
（このフォルダを ComfyUI-Gemini-Media2Text として配置）
..\..\python_embeded\python.exe -m pip install -r ComfyUI-Gemini-Media2Text\requirements.txt
```
ポータブル版以外は、ComfyUI を動かしている Python で `pip install -r requirements.txt` を実行する。
成功判定：ComfyUI 再起動後、ノード検索で「Gemini Media2Text (API Key)」が出る。

## API キーの設定（どちらか。①が優先）
1. 環境変数：`setx GEMINI_API_KEY "キー"` → 新しいコマンドプロンプトから ComfyUI を再起動
2. 設定ファイル：`ComfyUI\user\gemini_media2text\config.json` に `config.example.json` を参考に `api_key` を記入

キーはノードのウィジェットに入力しないため、ワークフロー JSON に残らない。

## 入力
| 名前 | 内容 |
|---|---|
| video / images | どちらか必須。images は1枚なら静止画、複数ならフレーム列（source_fps で元 fps を指定） |
| model | 起動時に API から取得。取得失敗時は固定リスト |
| prompt / system_prompt | prompt は必須 |
| fps | 送信 fps。元 fps 以上なら間引きなし。API の video_metadata.fps にも同値を渡す |
| total_megapixels | アスペクト比維持で縮小（拡大はしない） |
| h264_crf / jpeg_quality | 動画・静止画の品質 |
| media_resolution / thinking_level / max_output_tokens | API にそのまま渡す |
| include_audio | video 入力の音声を AAC で同梱 |
| range_unit / range_start / range_end | seconds か frames。end=0 は最後まで。ローカルで切り出してから送信 |
| cache_mode | normal / force_refresh（再取得して上書き）/ cache_only（キャッシュがなければ API を呼ばず失敗扱い） |
| on_block | passthrough（失敗時に fallback_text を出して続行）/ raise（停止。cache_miss では停止しない） |
| fallback_text | 失敗時に text へ流す文字列（任意入力） |

## 出力
text / success(BOOLEAN) / status（ok, cache_hit, blocked, cache_miss, error:…）/ usage(JSON) / cache_key

## キャッシュ
キー = 区間切り出し後・縮小前の画素（＋音声）のハッシュ ＋ モデル・プロンプト・各パラメータ。
秒指定とフレーム指定は同じ区間なら同じキーになる。
保存先：`ComfyUI\user\gemini_media2text\cache\<key>\`（text.txt / request.json / response.json）

## Files API
エンコード後 18MB 以上は Files API でアップロード。URI と期限を `files_index.json` に保存し、期限まで1時間以上あれば再利用。

## 概算料金
`config.json` の `prices`（USD / 100万トークン、output は思考トークン込み）で計算。
単価は改定されるため、公式料金ページで確認して書き換えること。
