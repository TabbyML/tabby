<div align="center">
  
# 🐾 Tabby

[📚 ドキュメント](https://tabby.tabbyml.com/docs/welcome/) • [💬 Slack](https://links.tabbyml.com/join-slack) • [🗺️ ロードマップ](https://tabby.tabbyml.com/docs/roadmap/)

[![最新リリース](https://shields.io/github/v/release/TabbyML/tabby)](https://github.com/TabbyML/tabby/releases/latest)
[![PR歓迎](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)
[![Docker pulls](https://img.shields.io/docker/pulls/tabbyml/tabby)](https://hub.docker.com/r/tabbyml/tabby)
[![codecov](https://codecov.io/gh/TabbyML/tabby/graph/badge.svg?token=WYVVH8MKK3)](https://codecov.io/gh/TabbyML/tabby)

[English](/README.md) |
[简体中文](/README-zh.md) |
[日本語](/README-ja.md)

</div>

Tabbyは、GitHub Copilotのオープンソースでオンプレミスな代替手段を提供する、セルフホスト型AIコーディングアシスタントです。いくつかの主要な特徴を備えています：
* DBMSやクラウドサービスが不要な自己完結型。
* OpenAPIインターフェースにより、既存のインフラストラクチャ（例：クラウドIDE）との統合が容易。
* コンシューマーグレードのGPUをサポート。

<p align="center">
  <a target="_blank" href="https://tabby.tabbyml.com"><img alt="ライブデモを開く" src="https://img.shields.io/badge/OPEN_LIVE_DEMO-blue?logo=xcode&style=for-the-badge&logoColor=green"></a>
</p>

<p align="center">
  <img alt="デモ" src="https://user-images.githubusercontent.com/388154/230440226-9bc01d05-9f57-478b-b04d-81184eba14ca.gif">
</p>

## 🔥 新着情報
* **2025/03/31** チャットサイドパネルにより豊富な`@`メニューを備えた[v0.27](https://github.com/TabbyML/tabby/releases/tag/v0.27.0)がリリースされました。
* **2025/02/05** LDAP認証とバックグラウンドジョブのより良い通知がTabby[v0.24.0](https://github.com/TabbyML/tabby/releases/tag/v0.24.0)に登場！✨
* **2025/02/04** [VSCode 1.20.0](https://marketplace.visualstudio.com/items/TabbyML.vscode-tabby/changelog)アップグレード！ファイルを@メンションしてチャットコンテキストに追加し、新しい右クリックオプションでインライン編集が可能に！
* **2025/01/10** Tabby[v0.23.0](https://github.com/TabbyML/tabby/releases/tag/v0.23.0)は、強化されたコードブラウザ体験とチャットサイドパネルの改善を特徴としています！

<details>
  <summary>アーカイブ</summary>
* **2024/12/24** Tabby[v0.22.0](https://github.com/TabbyML/tabby/releases/tag/v0.22.0)に**通知ボックス**を導入！
* **2024/12/06** Llamafileデプロイメント統合と強化されたアンサーエンジンユーザー体験がTabby[v0.21.0](https://github.com/TabbyML/tabby/releases/tag/v0.21.0)に登場！🚀
* **2024/11/10** 異なるバックエンドチャットモデル間の切り替えがTabby[v0.20.0](https://github.com/TabbyML/tabby/releases/tag/v0.20.0)のアンサーエンジンでサポートされました！
* **2024/10/30** Tabby[v0.19.0](https://github.com/TabbyML/tabby/releases/tag/v0.19.0)は、メインページに最近共有されたスレッドを表示し、その発見性を向上させます。
* **2024/07/09** 🎉[TabbyでのCodestral統合](https://tabby.tabbyml.com/blog/2024/07/09/tabby-codestral/)を発表！
* **2024/07/05** Tabby[v0.13.0](https://github.com/TabbyML/tabby/releases/tag/v0.13.0)は、内部エンジニアリングチームのための中央知識エンジンである***アンサーエンジン***を導入します。開発チームの内部データとシームレスに統合し、開発者に信頼性の高い正確な回答を提供します。
* **2024/06/13** [VSCode 1.7](https://marketplace.visualstudio.com/items/TabbyML.vscode-tabby/changelog)は、コーディング体験全体を通じて多用途なチャット体験を提供する重要なマイルストーンです。最新の**サイドパネルでのチャット**と**チャットコマンドによる編集**をお試しください！
* **2024/06/10** 最新の📃ブログ投稿がTabbyでの[強化されたコードコンテキスト理解](https://tabby.tabbyml.com/blog/2024/06/11/rank-fusion-in-tabby-code-completion/)について公開されました！
* **2024/06/06** Tabby[v0.12.0](https://github.com/TabbyML/tabby/releases/tag/v0.12.0)リリースは、🔗**シームレスな統合**（Gitlab SSO、セルフホストGitHub/GitLabなど）、⚙️**柔軟な設定**（HTTP API統合）、🌐**拡張された機能**（コードブラウザでのリポジトリコンテキスト）をもたらします。
* **2024/05/22** Tabby[VSCode 1.6](https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby)は、インライン補完での**複数の選択肢**と、**自動生成されたコミットメッセージ**🐱💻を提供します！
* **2024/05/11** [v0.11.0](https://github.com/TabbyML/tabby/releases/tag/v0.11.0)は、📊**ストレージ使用量**統計、🔗**GitHub & GitLab**統合、📋**アクティビティ**ページ、待望の🤖**Ask Tabby**機能を含む重要なエンタープライズアップグレードをもたらします！
* **2024/04/22** [v0.10.0](https://github.com/TabbyML/tabby/releases/tag/v0.10.0)がリリースされ、チームごとの分析を提供する最新の**レポート**タブを特徴としています。
* **2024/04/19** 📣 Tabbyは、コード補完のために[ローカルに関連するスニペット](https://github.com/TabbyML/tabby/pull/1844)（ローカルLSPからの宣言や最近変更されたコード）を組み込むようになりました！
* **2024/04/17** CodeGemmaとCodeQwenモデルシリーズが[公式レジストリ](https://tabby.tabbyml.com/docs/models/)に追加されました！
* **2024/03/20** [v0.9](https://github.com/TabbyML/tabby/releases/tag/v0.9.1)がリリースされ、フル機能の管理UIを強調しています。
* **2023/12/23** [SkyServe](https://skypilot.readthedocs.io/en/latest/serving/sky-serve.html) 🛫を使用して、[任意のクラウドでTabbyをシームレスにデプロイ](https://tabby.tabbyml.com/docs/installation/skypilot/)します。
* **2023/12/15** [v0.7.0](https://github.com/TabbyML/tabby/releases/tag/v0.7.0)がリリースされ、チーム管理と安全なアクセスを提供します！
* **2023/10/15** RAGベースのコード補完が[v0.3.0](https://github.com/TabbyML/tabby/releases/tag/v0.3.0)で詳細に有効化されました🎉！Tabbyがリポジトリレベルのコンテキストを利用してさらにスマートになる方法を説明する[ブログ投稿](https://tabby.tabbyml.com/blog/2023/10/16/repository-context-for-code-completion/)をチェックしてください！
* **2023/11/27** [v0.6.0](https://github.com/TabbyML/tabby/releases/tag/v0.6.0)がリリースされました！
* **2023/11/09** [v0.5.5](https://github.com/TabbyML/tabby/releases/tag/v0.5.5)がリリースされました！UIの再設計とパフォーマンスの向上を伴います。
* **2023/10/24** ⛳️ [VSCode/Vim/IntelliJ](https://tabby.tabbyml.com/docs/extensions)向けのTabby IDEプラグインの主要なアップデート！
* **2023/10/04** Tabbyがサポートする最新のモデルを確認するには、[モデルディレクトリ](https://tabby.tabbyml.com/docs/models/)をチェックしてください。
* **2023/09/18** AppleのM1/M2 Metal推論サポートが[v0.1.1](https://github.com/TabbyML/tabby/releases/tag/v0.1.1)に登場しました！
* **2023/08/31** Tabbyの最初の安定版リリース[v0.0.1](https://github.com/TabbyML/tabby/releases/tag/v0.0.1) 🥳。
* **2023/08/28** [CodeLlama 7B](https://github.com/TabbyML/tabby/issues/370)の実験的サポート。
* **2023/08/24** Tabbyが[JetBrains Marketplace](https://plugins.jetbrains.com/plugin/22379-tabby)に登場！

</details>

## 👋 はじめに

ドキュメントは[こちら](https://tabby.tabbyml.com/docs/getting-started)でご覧いただけます。
- 📚 [インストール](https://tabby.tabbyml.com/docs/installation/)
- 💻 [IDE/エディタ拡張](https://tabby.tabbyml.com/docs/extensions/)
- ⚙️ [設定](https://tabby.tabbyml.com/docs/configuration)

### 1分でTabbyを実行
Tabbyサーバーを開始する最も簡単な方法は、次のDockerコマンドを使用することです：

```bash
docker run -it \
  --gpus all -p 8080:8080 -v $HOME/.tabby:/data \
  tabbyml/tabby \
  serve --model StarCoder-1B --device cuda --chat-model Qwen2-1.5B-Instruct
```
追加のオプション（例：推論タイプ、並列処理）については、[ドキュメントページ](https://tabbyml.github.io/tabby)を参照してください。

## 🤝 コントリビューション

詳細なガイドは[CONTRIBUTING.md](https://github.com/TabbyML/tabby/blob/main/CONTRIBUTING.md)をご覧ください。

### コードを取得

```bash
git clone --recurse-submodules https://github.com/TabbyML/tabby
cd tabby
```

すでにリポジトリをクローンしている場合は、`git submodule update --recursive --init`コマンドを実行してすべてのサブモジュールを取得できます。

### ビルド

1. この[チュートリアル](https://www.rust-lang.org/learn/get-started)に従ってRust環境をセットアップします。

2. 必要な依存関係をインストールします：
```bash
# MacOSの場合
brew install protobuf

# Ubuntu / Debianの場合
apt install protobuf-compiler libopenblas-dev
```

3. 便利なツールをインストールします：
```bash
# Ubuntuの場合
apt install make sqlite3 graphviz
```

4. これで、`cargo build`コマンドを実行してTabbyをビルドできます。

### ハッキングを始めよう！
... そして、[プルリクエスト](https://github.com/TabbyML/tabby/compare)を提出するのを忘れないでください。

## 🌍 コミュニティ
- 🎤 [Twitter / X](https://twitter.com/Tabby_ML) - TabbyMLとあらゆる可能性について交流
- 📚 [LinkedIn](https://www.linkedin.com/company/tabbyml/) - コミュニティからの最新情報をフォロー
- 💌 [ニュースレター](https://newsletter.tabbyml.com/archive) - Tabbyの洞察と秘密を解き明かすために購読

### 🔆 アクティビティ

![Gitリポジトリアクティビティ](https://repobeats.axiom.co/api/embed/e4ef0fbd12e586ef9ea7d72d1fb4f5c5b88d78d5.svg "Repobeats分析画像")

### 🌟 スター履歴

[![スター履歴チャート](https://star-history.dera.page/svg?repos=tabbyml/tabby&type=Date)](https://star-history.dera.page/#tabbyml/tabby&Date)
