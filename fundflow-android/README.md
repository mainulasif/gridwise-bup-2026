# FundFlow v1.4 — Android app

This is a standard Android Gradle project, compiled by GitHub Actions. All UI and business logic is bundled in the single offline index.html file. The launcher is a conventional Java Activity hosting a WebView with local HTML loaded at an HTTPS base URL for localStorage support. No internet permission or backend is needed. This branch is not intended to be merged into GridWise; it only supplies temporary build infrastructure.

Build: `gradle -p fundflow-android :app:assembleDebug` with Android SDK 35 and JDK 17.
Install: Because the debug signing certificate differs from previous experimental APKs, uninstall old non-opening FundFlow before installing this one. Existing data inside the separate browser HTML file is unaffected.
