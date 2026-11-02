package com.fundflow.wallet;

import android.app.Activity;
import android.os.Bundle;
import android.graphics.Color;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.widget.TextView;
import android.view.ViewGroup;
import java.io.InputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

public final class MainActivity extends Activity {
    private WebView web;
    private static final int NIGHT = Color.rgb(11,16,32);

    @Override public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setStatusBarColor(NIGHT);
        getWindow().setNavigationBarColor(NIGHT);
        try {
            byte[] data;
            try (InputStream stream = getAssets().open("index.html");
                 ByteArrayOutputStream output = new ByteArrayOutputStream()) {
                byte[] buffer = new byte[8192];
                int n;
                while ((n = stream.read(buffer)) != -1) output.write(buffer, 0, n);
                data = output.toByteArray();
            }
            web = new WebView(this);
            web.setBackgroundColor(NIGHT);
            WebSettings settings = web.getSettings();
            settings.setJavaScriptEnabled(true);
            settings.setDomStorageEnabled(true);
            settings.setAllowFileAccess(false);
            settings.setAllowContentAccess(false);
            settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
            web.setWebChromeClient(new WebChromeClient());
            web.setWebViewClient(new WebViewClient() {
               @Override public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
                  if (request.isForMainFrame()) showError("WebView failed: " + error.getDescription());
               }
            });
            setContentView(web, new ViewGroup.LayoutParams(-1,-1));
            web.loadDataWithBaseURL("https://fundflow.local/", new String(data, StandardCharsets.UTF_8), "text/html", "UTF-8", null);
        } catch (Throwable ex) {
            showError("Could not initialize FundFlow: " + ex.getClass().getSimpleName() + ": " + ex.getMessage());
        }
    }
    private void showError(String message) {
        TextView error = new TextView(this);
        error.setText("FundFlow\n\n" + message);
        error.setTextSize(17);
        error.setTextColor(Color.WHITE);
        error.setBackgroundColor(NIGHT);
        error.setPadding(30,80,30,30);
        setContentView(error);
    }
    @Override public void onBackPressed() {
        if (web != null && web.canGoBack()) web.goBack();
        else super.onBackPressed();
    }
    @Override public void onDestroy() {
        if (web != null) { ((ViewGroup)web.getParent()).removeView(web); web.destroy(); web = null; }
        super.onDestroy();
    }
}