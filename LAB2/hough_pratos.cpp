/**
 * Laboratório 02 — Detecção de Pratos com Transformada de Hough
 * UC de Visão Computacional | Prof. Dr. Fábio Cappabianco
 *
 * Compilar:
 *   cmake -S . -B build && cmake --build build
 *
 * Uso:
 *   ./build/hough_pratos <pasta_entrada> <pasta_saida> [config.ini]
 *   Exemplo: ./build/hough_pratos ./imagens ./saida config.ini
 *
 * Os parâmetros são lidos de config.ini — edite-o sem precisar recompilar.
 */

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;
using namespace cv;

// ═══════════════════════════════════════════════════════════════════════════════
// Leitor de arquivo .ini simples
// ═══════════════════════════════════════════════════════════════════════════════
class IniConfig {
public:
    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;

        std::string section, line;
        while (std::getline(f, line)) {
            // Remove comentários e espaços
            auto pos = line.find('#');
            if (pos != std::string::npos) line = line.substr(0, pos);
            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            if (line.empty()) continue;

            if (line.front() == '[' && line.back() == ']') {
                section = line.substr(1, line.size() - 2);
            } else {
                auto eq = line.find('=');
                if (eq == std::string::npos) continue;
                std::string key = line.substr(0, eq);
                std::string val = line.substr(eq + 1);
                key.erase(key.find_last_not_of(" \t") + 1);
                val.erase(0, val.find_first_not_of(" \t"));
                data[section + "." + key] = val;
            }
        }
        return true;
    }

    double getDouble(const std::string& sec, const std::string& key, double def = 0.0) const {
        auto it = data.find(sec + "." + key);
        return (it != data.end()) ? std::stod(it->second) : def;
    }

    int getInt(const std::string& sec, const std::string& key, int def = 0) const {
        auto it = data.find(sec + "." + key);
        return (it != data.end()) ? std::stoi(it->second) : def;
    }

    bool getBool(const std::string& sec, const std::string& key, bool def = false) const {
        return getInt(sec, key, def ? 1 : 0) != 0;
    }

    void print() const {
        std::cout << "\n┌─ Configuração carregada ─────────────────────────\n";
        for (const auto& [k, v] : data)
            std::cout << "│  " << k << " = " << v << "\n";
        std::cout << "└──────────────────────────────────────────────────\n\n";
    }

private:
    std::map<std::string, std::string> data;
};

// ═══════════════════════════════════════════════════════════════════════════════
// Pré-processamento
// ═══════════════════════════════════════════════════════════════════════════════
Mat preprocessar(const Mat& src, const IniConfig& cfg) {
    Mat gray, result;

    cvtColor(src, gray, COLOR_BGR2GRAY);
    result = gray.clone();

    // Redimensionamento (se configurado)
    int rw = cfg.getInt("output", "resize_width");
    if (rw > 0) {
        double scale = (double)rw / src.cols;
        resize(result, result, Size(), scale, scale, INTER_AREA);
    }

    // Filtro Gaussiano
    int gk = cfg.getInt("preprocessing", "gaussian_ksize");
    if (gk > 0) {
        if (gk % 2 == 0) gk++;
        double gs = cfg.getDouble("preprocessing", "gaussian_sigma", 2.0);
        GaussianBlur(result, result, Size(gk, gk), gs);
    }

    // Filtro da Mediana
    int mk = cfg.getInt("preprocessing", "median_ksize");
    if (mk > 0) {
        if (mk % 2 == 0) mk++;
        medianBlur(result, result, mk);
    }

    // Equalização de histograma global
    if (cfg.getBool("preprocessing", "equalize_hist")) {
        equalizeHist(result, result);
    }

    // CLAHE (tem prioridade sobre equalização global se ambos ativos)
    if (cfg.getBool("preprocessing", "clahe")) {
        double clip = cfg.getDouble("preprocessing", "clahe_clip", 2.0);
        int    tile = cfg.getInt("preprocessing",   "clahe_tile",  8);
        auto clahe  = createCLAHE(clip, Size(tile, tile));
        clahe->apply(result, result);
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Detecção
// ═══════════════════════════════════════════════════════════════════════════════
std::vector<Vec3f> detectarCirculos(const Mat& processed, const IniConfig& cfg) {
    std::vector<Vec3f> circles;
    HoughCircles(
        processed,
        circles,
        HOUGH_GRADIENT,
        cfg.getDouble("hough", "dp",        1.5),
        cfg.getDouble("hough", "minDist",   100),
        cfg.getDouble("hough", "param1",    100),
        cfg.getDouble("hough", "param2",     40),
        cfg.getInt   ("hough", "minRadius",  80),
        cfg.getInt   ("hough", "maxRadius",   0)
    );

    int maxC = cfg.getInt("output", "max_circles");
    if (maxC > 0 && (int)circles.size() > maxC)
        circles.resize(maxC);

    return circles;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Desenho
// ═══════════════════════════════════════════════════════════════════════════════
Mat desenharCirculos(const Mat& src, const std::vector<Vec3f>& circles) {
    Mat output = src.clone();
    for (size_t i = 0; i < circles.size(); ++i) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int   radius = cvRound(circles[i][2]);
        circle(output, center, radius, Scalar(0, 255, 0), 3);
        circle(output, center, 4,      Scalar(0, 0, 255), -1);
        putText(output, std::to_string(i + 1), center + Point(6, -6),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 200, 255), 2);
    }
    std::string info = "Circulos: " + std::to_string(circles.size());
    putText(output, info, Point(10, 35),
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 0), 2);
    return output;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0]
                  << " <pasta_entrada> <pasta_saida> [config.ini]\n";
        return 1;
    }

    fs::path inputDir  = argv[1];
    fs::path outputDir = argv[2];
    std::string configPath = (argc >= 4) ? argv[3] : "config.ini";

    IniConfig cfg;
    if (!cfg.load(configPath)) {
        std::cerr << "Aviso: nao foi possivel abrir '" << configPath
                  << "'. Usando valores padrao.\n";
    } else {
        std::cout << "Config carregada de: " << configPath;
        cfg.print();
    }

    if (!fs::exists(outputDir))
        fs::create_directories(outputDir);

    // Subpasta para imagens pré-processadas
    fs::path prepDir = outputDir / "preprocessado";
    bool salvarPrep = cfg.getBool("output", "save_preprocessed", true);
    if (salvarPrep && !fs::exists(prepDir))
        fs::create_directories(prepDir);

    const std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp"};

    int total = 0, semDeteccao = 0;
    std::ofstream log(outputDir / "resultados.csv");
    log << "arquivo,circulos_detectados\n";

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        bool ehImagem = std::any_of(exts.begin(), exts.end(),
                                    [&](const std::string& e){ return e == ext; });
        if (!ehImagem) continue;

        Mat src = imread(entry.path().string());
        if (src.empty()) {
            std::cerr << "Erro ao ler: " << entry.path() << "\n";
            continue;
        }

        Mat processed = preprocessar(src, cfg);

        // Salvar imagem pre-processada (grayscale) antes do Hough
        if (salvarPrep) {
            fs::path prepPath = prepDir / entry.path().filename();
            imwrite(prepPath.string(), processed);
        }

        auto circles  = detectarCirculos(processed, cfg);
        Mat resultado = desenharCirculos(src, circles);

        fs::path outPath = outputDir / entry.path().filename();
        imwrite(outPath.string(), resultado);

        log << entry.path().filename().string() << ","
            << circles.size() << "\n";

        if (circles.empty()) ++semDeteccao;
        ++total;

        std::cout << "[" << total << "] "
                  << entry.path().filename().string()
                  << " -> " << circles.size() << " circulo(s)\n";
    }

    log.close();

    std::cout << "\n==========================================\n";
    std::cout << "Total processado : " << total                  << "\n";
    std::cout << "Com deteccao     : " << (total - semDeteccao)  << "\n";
    std::cout << "Sem deteccao     : " << semDeteccao            << "\n";
    std::cout << "==========================================\n";
    std::cout << "Saida : " << outputDir << "\n";
    std::cout << "CSV   : " << (outputDir / "resultados.csv") << "\n";

    return 0;
}