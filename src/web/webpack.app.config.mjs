import path from "path";
import { fileURLToPath } from "url";
import HtmlWebpackPlugin from "html-webpack-plugin";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default {
    entry: "./app/src/index.ts",
    resolve: { extensions: [".ts", ".js"] },
    module: {
        rules: [
            {
                test: /\.ts$/,
                use: {
                    loader: "ts-loader",
                    options: {
                        configFile: "app/tsconfig.json",
                        transpileOnly: true,
                    },
                },
                exclude: /node_modules/,
            },
            {
                test: /\.css$/,
                use: ["style-loader", "css-loader"],
            },
            {
                test: /\.(woff2?|ttf|otf)$/i,
                type: "asset/resource",
                generator: {
                    filename: "fonts/[name].[contenthash][ext]",
                },
            },
            {
                test: /\.(png|jpe?g|webp)$/i,
                type: "asset/resource",
                generator: {
                    filename: "images/[name].[contenthash][ext]",
                },
            },
            {
                test: /\.worklet\.js$/,
                type: "asset/resource",
                generator: {
                    filename: "worklets/[name].[contenthash][ext]",
                },
            },
        ],
    },
    plugins: [
        new HtmlWebpackPlugin({
            template: "./app/index.html",
            title: "X-Live",
        }),
    ],
    output: {
        path: path.resolve(__dirname, "app-dist"),
        filename: "assets/[name].[contenthash].js",
        clean: true,
        publicPath: "./",
    },
    devServer: {
        host: "0.0.0.0",
        port: 8000,
        // Cluster web-service proxies replace the Host header with a generated
        // public hostname. This development server only serves the static UI.
        allowedHosts: "all",
        proxy: [
            {
                context: ["/xlive-ws", "/api", "/healthz"],
                pathRewrite: { "^/xlive-ws": "/ws" },
                target: "http://127.0.0.1:8001",
                ws: true,
                changeOrigin: true,
            },
        ],
        hot: false,
        liveReload: false,
        client: false,
        historyApiFallback: true,
    },
    devtool: "source-map",
    stats: "errors-warnings",
};
