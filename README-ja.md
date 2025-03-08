<div align="center">

# 🐾 Tabby

[📚 ドキュメント](https://tabby.tabbyml.com/docs/welcome/) • [💬 Slack](https://links.tabbyml.com/join-slack) • [🗺️ ロードマップ](https://tabby.tabbyml.com/docs/roadmap/)

[![最新リリース](https://shields.io/github/v/release/TabbyML/tabby)](https://github.com/TabbyML/tabby/releases/latest)
[![PR歓迎](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)
[![Docker pulls](https://img.shields.io/docker/pulls/tabbyml/tabby)](https://hub.docker.com/r/tabbyml/tabby)
[![codecov](https://codecov.io/gh/TabbyML/tabby/graph/badge.svg?token=WYVVH8MKK3)](https://codecov.io/gh/TabbyML/tabby)

</div>

Tabbyは、GitHub Copilotのオープンソースかつオンプレミスの代替として提供されるセルフホスト型AIコーディングアシスタントです。以下の主要な特徴を備えています：
* DBMSやクラウドサービスが不要な自己完結型。
* OpenAPIインターフェースにより、既存のインフラ（例：クラウドIDE）と簡単に統合可能。
* コンシューマー向けGPUをサポート。

<p align="center">
  <a target="_blank" href="https://tabby.tabbyml.com"><img alt="ライブデモを開く" src="https://img.shields.io/badge/OPEN_LIVE_DEMO-blue?logo=xcode&style=for-the-badge&logoColor=green"></a>
</p>

<p align="center">
  <img alt="デモ" src="https://user-images.githubusercontent.com/388154/230440226-9bc01d05-9f57-478b-b04d-81184eba14ca.gif">
</p>

## 🔥 新着情報
* **2025年2月5日** LDAP認証とバックグラウンドジョブのより良い通知がTabby [v0.24.0](https://github.com/TabbyML/tabby/releases/tag/v0.24.0)に登場！✨
* **2025年2月4日** [VSCode 1.20.0](https://marketplace.visualstudio.com/items/TabbyML.vscode-tabby/changelog) アップグレード！ファイルをチャットコンテキストとして追加するための@メンションや、新しい右クリックオプションでのインライン編集が可能に！
* **2025年1月10日** Tabby [v0.23.0](https://github.com/TabbyML/tabby/releases/tag/v0.23.0) では、コードブラウザ体験の向上とチャットサイドパネルの改善を特徴としています！

<details>
  <summary>アーカイブ</summary>
* **2024年12月24日** Tabby [v0.22.0](https://github.com/TabbyML/tabby/releases/tag/v0.22.0)に**通知ボックス**を導入！
* **2024年12月6日** Llamafileデプロイメント統合と強化されたアンサーエンジンユーザー体験がTabby [v0.21.0](https://github.com/TabbyML/tabby/releases/tag/v0.21.0)に登場！🚀
* **2024年11月10日** Tabby [v0.20.0](https://github.com/TabbyML/tabby/releases/tag/v0.20.0) では、アンサーエンジンで異なるバックエンドチャットモデル間の切り替えがサポートされます！
* **2024年10月30日** Tabby [v0.19.0](https://github.com/TabbyML/tabby/releases/tag/v0.19.0) では、メインページに最近共有されたスレッドを表示し、発見しやすくしています。
* **2024年7月9日** 🎉 [Codestral統合をTabbyで発表](https://tabby.tabbyml.com/blog/2024/07/09/tabby-codestral/)！
* **2024年7月5日** Tabby [v0.13.0](https://github.com/TabbyML/tabby/releases/tag/v0.13.0) は、内部エンジニアリングチームのための中央知識エンジンである***アンサーエンジン***を導入。開発チームの内部データとシームレスに統合し、開発者に信頼性の高い正確な回答を提供します。
* **2024年6月13日** [VSCode 1.7](https://marketplace.visualstudio.com/items/TabbyML.vscode-tabby/changelog) は、コーディング体験全体を通じて多用途なチャット体験を提供する重要なマイルストーンです。最新の**サイドパネルでのチャット**や**チャットコマンドによる編集**をお試しください！
* **2024年6月10日** 最新の📃ブログ投稿がTabbyでの[強化されたコードコンテキスト理解](https://tabby.tabbyml.com/blog/2024/06/11/rank-fusion-in-tabby-code-completion/)について公開されました！
* **2024年6月6日** Tabby [v0.12.0](https://github.com/TabbyML/tabby/releases/tag/v0.12.0) リリースでは、🔗**シームレスな統合**（Gitlab SSO、セルフホストGitHub/GitLabなど）、⚙️**柔軟な設定**（HTTP API統合）、🌐**拡張された機能**（コードブラウザでのリポジトリコンテキスト）を提供します。
* **2024年5月22日** Tabby [VSCode 1.6](https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby) は、インライン補完での**複数の選択肢**や、**自動生成されたコミットメッセージ**🐱💻を提供します！
* **2024年5月11日** [v0.11.0](https://github.com/TabbyML/tabby/releases/tag/v0.11.0) は、📊**ストレージ使用量**の統計、🔗**GitHub & GitLab**の統合、📋**アクティビティ**ページ、待望の🤖**Ask Tabby**機能を含む重要なエンタープライズアップグレードをもたらします！
* **2024年4月22日** [v0.10.0](https://github.com/TabbyML/tabby/releases/tag/v0.10.0) がリリースされ、Tabby使用のチーム別分析を行う最新の**レポート**タブを特徴としています。
* **2024年4月19日** 📣 Tabbyは、コード補完のための[ローカルに関連するスニペット](https://github.com/TabbyML/tabby/pull/1844)（ローカルLSPからの宣言や最近変更されたコード）を組み込みました！
* **2024年4月17日** CodeGemmaとCodeQwenモデルシリーズが[公式レジストリ](https://tabby.tabbyml.com/docs/models/)に追加されました！
* **2024年3月20日** [v0.9](https://github.com/TabbyML/tabby/releases/tag/v0.9.1) がリリースされ、フル機能の管理UIを強調しています。
* **2023年12月23日** [SkyServe](https://skypilot.readthedocs.io/en/latest/serving/sky-serve.html) 🛫を使用して、[任意のクラウドにTabbyをシームレスにデプロイ](https://tabby.tabbyml.com/docs/installation/skypilot/)。
* **2023年12月15日** [v0.7.0](https://github.com/TabbyML/tabby/releases/tag/v0.7.0) がチーム管理と安全なアクセスを備えてリリースされました！
* **2023年10月15日** RAGベースのコード補完が[v0.3.0](https://github.com/TabbyML/tabby/releases/tag/v0.3.0)🎉で有効になりました！Tabbyがリポジトリレベルのコンテキストを利用してさらに賢くなる方法を説明する[ブログ投稿](https://tabby.tabbyml.com/blog/2023/10/16/repository-context-for-code-completion/)をチェックしてください！
* **2023年11月27日** [v0.6.0](https://github.com/TabbyML/tabby/releases/tag/v0.6.0) がリリースされました！
* **2023年11月9日** [v0.5.5](https://github.com/TabbyML/tabby/releases/tag/v0.5.5) がリリースされました！UIの再設計とパフォーマンスの向上を伴います。
* **2023年10月24日** ⛳️ [VSCode/Vim/IntelliJ](https://tabby.tabbyml.com/docs/extensions) 向けのTabby IDEプラグインの大規模アップデート！
* **2023年10月4日** Tabbyがサポートする最新のモデルを確認するには、[モデルディレクトリ](https://tabby.tabbyml.com/docs/models/)をチェックしてください。
* **2023年9月18日** AppleのM1/M2 Metal推論サポートが[v0.1.1](https://github.com/TabbyML/tabby/releases/tag/v0.1.1)に登場！
* **2023年8月31日** Tabbyの最初の安定版リリース[v0.0.1](https://github.com/TabbyML/tabby/releases/tag/v0.0.1) 🥳。
* **2023年8月28日** [CodeLlama 7B](https://github.com/TabbyML/tabby/issues/370) の実験的サポート。
* **2023年8月24日** Tabbyが[JetBrains Marketplace](https://plugins.jetbrains.com/plugin/22379-tabby)に登場！

</details>

## 👋 はじめに

ドキュメントは[こちら](https://tabby.tabbyml.com/docs/getting-started)でご覧いただけます。
- 📚 [インストール](https://tabby.tabbyml.com/docs/installation/)
- 💻 [IDE/エディタ拡張](https://tabby.tabbyml.com/docs/extensions/)
- ⚙️ [設定](https://tabby.tabbyml.com/docs/configuration)

### 1分でTabbyを実行
Tabbyサーバーを開始する最も簡単な方法は、以下のDockerコマンドを使用することです：

```bash
docker run -it \
  --gpus all -p 8080:8080 -v $HOME/.tabby:/data \
  tabbyml/tabby \
  serve --model StarCoder-1B --device cuda --chat-model Qwen2-1.5B-Instruct
```
追加のオプション（例：推論タイプ、並列処理）については、[ドキュメントページ](https://tabbyml.github.io/tabby)を参照してください。

## 🤝 コントリビューション

詳細なガイドは[CONTRIBUTING.md](https://github.com/TabbyML/tabby/blob/main/CONTRIBUTING.md)でご覧いただけます。

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

### ハッキングを始めましょう！
...そして、[プルリクエスト](https://github.com/TabbyML/tabby/compare)を忘れずに提出してください。

## 🌍 コミュニティ
- 🎤 [Twitter / X](https://twitter.com/Tabby_ML) - TabbyMLに関するすべてのことに関与する
- 📚 [LinkedIn](https://www.linkedin.com/company/tabbyml/) - コミュニティからの最新情報をフォローする
- 💌 [ニュースレター](https://newsletter.tabbyml.com/archive) - Tabbyの洞察と秘密を解き明かすために購読する

### 🔆 アクティビティ

![Gitリポジトリアクティビティ](https://repobeats.axiom.co/api/embed/e4ef0fbd12e586ef9ea7d72d1fb4f5c5b88d78d5.svg "Repobeats analytics image")

### 🌟 スター履歴

[![スター履歴チャート](https://api.star-history.com/svg?repos=tabbyml/tabby&type=Date)](https://star-history.com/#tabbyml/tabby&Date)