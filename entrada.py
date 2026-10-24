"""Inspeção e seleção explícita de colunas, sem treinar ou gravar arquivos.

As posições x1..xN pertencem ao arquivo original. Descrições são somente
anotações; ID e alvo nunca são características da rede.
"""
from __future__ import annotations

from collections import Counter
import csv
import io
import math
import re
import unicodedata


_BASE_FEATURES = ["radius", "texture", "perimeter", "area", "smoothness",
                  "compactness", "concavity", "concave_points", "symmetry",
                  "fractal_dimension"]
WDBC_FEATURES = [f"{name}_{group}" for group in ("mean", "se", "worst")
                 for name in _BASE_FEATURES]
_ID_NAMES = {"id", "identificador", "codigo_id"}
_TARGET_NAMES = {"diagnosis", "diagnostico", "classe", "class", "target", "label",
                 "sex", "sexo", "gender", "genero", "species", "especie", "cultivar"}
_ROLES = {"feature", "id", "target", "ignore"}
_DELIMITERS = {",": ",", ";": ";", "\t": "\t", "comma": ",",
               "semicolon": ";", "tab": "\t", "whitespace": "whitespace"}
PROTEIN_FEATURES = ["mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc"]
_PROTEIN_CLASSES = ["CYT", "NUC", "MIT", "ME3", "ME2", "ME1", "EXC", "VAC", "POX", "ERL"]
_PROTEIN_DESCRIPTIONS = [
    "ID da sequência: identificador de acesso da proteína no SWISS-PROT.",
    "mcg: escore do método de McGeoch para reconhecer sequências de sinal.",
    "gvh: escore do método de von Heijne para reconhecer sequências de sinal.",
    "alm: escore do ALOM para prever regiões que atravessam a membrana.",
    "mit: escore da composição dos 20 primeiros aminoácidos para localização mitocondrial.",
    "erl: atributo descrito como binário para a presença do sinal HDEL de retenção no retículo endoplasmático; valores do arquivo preservados.",
    "pox: sinal de direcionamento ao peroxissomo na extremidade C-terminal.",
    "vac: escore da composição de aminoácidos para proteínas vacuolares e extracelulares.",
    "nuc: escore de sinais de localização nuclear para proteínas nucleares e não nucleares.",
    "Classe: local de localização da proteína; é o alvo e não uma entrada da rede.",
]
ABALONE_FEATURES = ["length", "diameter", "height", "whole_weight", "shucked_weight",
                    "viscera_weight", "shell_weight", "rings"]
IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
WINE_FEATURES = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
                 "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
                 "color_intensity", "hue", "od280_od315", "proline"]
TABULAR_PROFILE_TITLES = {
    "1. Title of Database: Abalone data": "abalone_gender",
    "1. Title: Iris Plants Database": "iris",
    "1. Title of Database: Wine recognition data": "wine",
}
_TABULAR_PROFILES = {
    "abalone_gender": {
        "label": "Abalone (M/F/I)", "names": ["sex", *ABALONE_FEATURES], "target": 0,
        "descriptions": [
            "Classe M/F/I: M = masculino, F = feminino e I = infantil; alvo escolhido para esta base.",
            "Length: maior comprimento da concha (mm na documentação original).",
            "Diameter: diâmetro perpendicular ao comprimento (mm na documentação original).",
            "Height: altura com a carne na concha (mm na documentação original).",
            "Whole weight: peso do abalone inteiro (gramas na documentação original).",
            "Shucked weight: peso da carne (gramas na documentação original).",
            "Viscera weight: peso das vísceras após sangria (gramas na documentação original).",
            "Shell weight: peso da concha depois de seca (gramas na documentação original).",
            "Rings: número de anéis; anéis + 1,5 estima a idade em anos. Entrada para prever M/F/I.",
        ],
    },
    "iris": {
        "label": "Iris", "names": [*IRIS_FEATURES, "species"], "target": 4,
        "descriptions": ["Comprimento da sépala (cm).", "Largura da sépala (cm).",
                         "Comprimento da pétala (cm).", "Largura da pétala (cm).",
                         "Espécie da flor Iris; é o alvo e não uma entrada da rede."],
    },
    "wine": {
        "label": "Wine", "names": ["classe", *WINE_FEATURES], "target": 0,
        "descriptions": [
            "Classe do vinho (1, 2 ou 3): identifica o cultivar; alvo, não ID da amostra.",
            "Alcohol: teor de álcool.", "Malic acid: ácido málico.", "Ash: cinzas.",
            "Alcalinity of ash: alcalinidade das cinzas.", "Magnesium: magnésio.",
            "Total phenols: fenóis totais.", "Flavanoids: flavonoides.",
            "Nonflavanoid phenols: fenóis não flavonoides.", "Proanthocyanins: proantocianinas.",
            "Color intensity: intensidade da cor.", "Hue: tonalidade.",
            "OD280/OD315 of diluted wines: razão OD280/OD315 de vinhos diluídos.", "Proline: prolina.",
        ],
    },
}


def _name_key(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.strip().lower())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[\s-]+", "_", value)


def _number(value: str, delimiter: str) -> float:
    value = value.strip()
    if not value:
        raise ValueError("valor vazio")
    if delimiter != "," and "," in value:
        if "." in value or value.count(",") != 1:
            raise ValueError("separador decimal ambíguo")
        value = value.replace(",", ".")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("valor não finito")
    return number


def _is_number(value: str, delimiter: str) -> bool:
    try:
        _number(value, delimiter)
        return True
    except (ValueError, OverflowError):
        return False


def _records(text: str, delimiter: str, offset: int = 0):
    if delimiter == "whitespace":
        # Cada linha física é um registro; nunca procurar a próxima linha
        # válida para contornar uma amostra com colunas faltantes.
        records = [(line.split(), number + offset)
                   for number, line in enumerate(text.splitlines(), 1) if line.strip()]
        return [row for row, _ in records], [line for _, line in records]
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    rows, source_lines = [], []
    previous_end = 0
    try:
        for row in reader:
            start = previous_end + 1 + offset
            previous_end = reader.line_num
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            rows.append(row)
            source_lines.append(start)
    except csv.Error as exc:
        raise ValueError(f"CSV inválido próximo da linha {reader.line_num + offset}: {exc}.") from exc
    return rows, source_lines


def _wdbc_payload(text: str):
    """Remove somente a documentação do TXT WDBC reconhecido, nunca texto genérico."""
    lines = text.splitlines(keepends=True)
    first = next((line.strip() for line in lines if line.strip()), "")
    if first != "1. Title: Wisconsin Diagnostic Breast Cancer (WDBC)":
        return text, 0, False
    attribute_marker = re.compile(
        r"^6\. Number of attributes:\s*32\s*\(ID, diagnosis, 30 real-valued input features\)\s*$")
    class_marker = re.compile(r"^9\. Class distribution:\s*357 benign,\s*212 malignant\s*$")
    if not any(attribute_marker.match(line.strip()) for line in lines):
        raise ValueError("Preâmbulo WDBC incompleto ou não reconhecido: esperadas 32 colunas.")
    marker = next((i for i, line in enumerate(lines) if class_marker.match(line.strip())), None)
    if marker is None:
        raise ValueError("Preâmbulo WDBC incompleto: início dos registros não encontrado.")
    return "".join(lines[marker + 1:]), marker + 1, True


def _protein_payload(text: str):
    """Reconhece a documentação de Yeast; não remove introduções genéricas."""
    lines = text.splitlines(keepends=True)
    first = next((line.strip() for line in lines if line.strip()), "")
    if first != "1. Title: Protein Localization Sites":
        return text, 0, False
    attributes = re.compile(r"^for Yeast dataset:\s*9\s*\(\s*8 predictive,\s*1 name\s*\)\s*$")
    if not any(attributes.match(line.strip()) for line in lines):
        raise ValueError("Preâmbulo Protein Localization Sites incompleto: esperados ID e 8 medidas Yeast.")
    for number, name in enumerate(["Sequence Name", *PROTEIN_FEATURES], 1):
        marker = re.compile(rf"^{number}\.\s+{re.escape(name)}:")
        if not any(marker.match(line.strip()) for line in lines):
            raise ValueError(f"Preâmbulo Protein Localization Sites incompleto: atributo {name} não encontrado.")
    distribution = next((i for i, line in enumerate(lines)
                         if line.strip().startswith("9. Class Distribution. The class is the localization site.")), None)
    if distribution is None:
        raise ValueError("Preâmbulo Protein Localization Sites incompleto: legenda das classes não encontrada.")
    legend = re.compile(r"^([A-Z0-9]{3})\s+\([^()]+\)\s+\d+[A-Za-z*]?\s*$")
    seen = []
    end = None
    for index in range(distribution + 1, len(lines)):
        match = legend.match(lines[index].strip())
        if match:
            seen.append(match.group(1))
            if match.group(1) == "ERL":
                end = index
                break
    if seen != _PROTEIN_CLASSES or end is None:
        raise ValueError("Preâmbulo Protein Localization Sites incompleto: esperada a legenda das 10 classes Yeast até ERL.")
    # O limite é a legenda, nunca o aspecto de uma amostra. Uma primeira
    # amostra truncada continua sendo lida e causa erro com sua linha real.
    return "".join(lines[end + 1:]), end + 1, True


def _tabular_profile_payload(text: str):
    """Separa três preâmbulos conhecidos por seus marcadores documentais.

    O limite nunca depende de achar uma amostra válida: a primeira linha
    depois da documentação pertence aos dados, mesmo quando malformada.
    Quantidades da legenda não substituem a contagem dos registros reais.
    """
    lines = text.splitlines(keepends=True)
    first = next((line.strip() for line in lines if line.strip()), "")
    profile_name = TABULAR_PROFILE_TITLES.get(first)
    if profile_name is None:
        return text, 0, None
    label = _TABULAR_PROFILES[profile_name]["label"]

    def require(pattern, description):
        found = next((i for i, line in enumerate(lines) if re.fullmatch(pattern, line.strip())), None)
        if found is None:
            raise ValueError(f"Preâmbulo {label} incompleto: {description}.")
        return found

    if profile_name == "iris":
        require(r"6\. Number of Attributes:\s*4 numeric, predictive attributes and the class",
                "esperadas quatro medidas e a classe")
        for number, name in enumerate(["sepal length", "sepal width", "petal length", "petal width"], 1):
            require(rf"{number}\.\s+{name}\s+in cm", f"atributo {name} não encontrado")
        require(r"5\.\s+class:\s*", "coluna da classe não encontrada")
        end = require(r"9\. Class Distribution:\s*33\.3% for each of 3 classes\.",
                      "fim da distribuição das três classes não encontrado")
    elif profile_name == "wine":
        attributes = require(r"6\. Number of Attributes", "seção dos 13 atributos não encontrada")
        following = next((line.strip() for line in lines[attributes + 1:] if line.strip()), "")
        if following != "13":
            raise ValueError("Preâmbulo Wine incompleto: esperados 13 atributos numéricos.")
        names = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
                 "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
                 "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
        for number, name in enumerate(names, 1):
            require(rf"{number}\)\s*{re.escape(name)}", f"atributo {name} não encontrado")
        require(r"NOTE:\s*1st attribute is class identifier \(1-3\)",
                "posição da classe não encontrada")
        distribution = require(r"9\. Class Distribution: number of instances per class",
                               "distribuição das classes não encontrada")
        expected = 1
        end = None
        for index in range(distribution + 1, len(lines)):
            line = lines[index].strip()
            if not line:
                continue
            if not re.fullmatch(rf"class\s+{expected}\s+\d+", line):
                raise ValueError(f"Preâmbulo Wine incompleto: esperada legenda da classe {expected}.")
            if expected == 3:
                end = index
                break
            expected += 1
        if end is None:
            raise ValueError("Preâmbulo Wine incompleto: esperada legenda das três classes.")
    else:
        require(r"6\. Number of Attributes:\s*8", "esperados sexo e sete medidas, além de Rings")
        names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight",
                 "Viscera weight", "Shell weight", "Rings"]
        for name in names:
            kind = "nominal" if name == "Sex" else "integer" if name == "Rings" else "continuous"
            require(rf"{re.escape(name)}\s+{kind}(?:\s+.*)?", f"atributo {name} não encontrado")
        distribution = require(r"9\. Class Distribution:", "distribuição de anéis não encontrada")
        header_seen, counts_seen, end = False, False, None
        for index in range(distribution + 1, len(lines)):
            line = lines[index].strip()
            if not line or re.fullmatch(r"-+\s+-+", line):
                continue
            if not header_seen and re.fullmatch(r"Class\s+Examples", line):
                header_seen = True
                continue
            if header_seen and re.fullmatch(r"\d+\s+\d+", line):
                counts_seen = True
                continue
            if counts_seen and re.fullmatch(r"Total\s+\d+", line):
                end = index
                break
            raise ValueError("Preâmbulo Abalone (M/F/I) incompleto: legenda dos anéis não reconhecida.")
        if end is None:
            raise ValueError("Preâmbulo Abalone (M/F/I) incompleto: total da legenda não encontrado.")
    return "".join(lines[end + 1:]), end + 1, profile_name


def _detect_delimiter(text: str) -> str:
    # A largura da primeira linha evita que uma linha incompleta seja escondida
    # pela escolha de um separador que transforma a tabela em uma coluna só.
    choices, malformed = [], []
    for priority, delimiter in enumerate((",", ";", "\t")):
        try:
            rows, _ = _records(text, delimiter)
        except ValueError as exc:
            # Não reinterpretar um CSV com aspas quebradas como uma coluna só.
            try:
                reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=False)
                first_row = next(row for row in reader
                                 if row and not (len(row) == 1 and not row[0].strip()))
                if len(first_row) > 1:
                    malformed.append((len(first_row), str(exc)))
            except (StopIteration, csv.Error):
                pass
            continue
        if rows:
            widths = Counter(map(len, rows))
            first_width = len(rows[0])
            modal_width, frequency = widths.most_common(1)[0]
            choices.append((min(first_width, modal_width) > 1,
                            frequency / len(rows) if modal_width > 1 else 0,
                            first_width, -priority, delimiter))
    if malformed and (not choices or not max(choices)[0]):
        raise ValueError(max(malformed)[1])
    if not choices:
        raise ValueError("Não foi possível ler o arquivo CSV/TXT/TSV.")
    if not max(choices)[0]:
        rows, _ = _records(text, "whitespace")
        if rows and len(rows[0]) > 1:
            return "whitespace"
    return max(choices)[-1]


def _has_header(rows: list[list[str]], delimiter: str) -> bool:
    first = rows[0]
    # Um ID textual pode ser literalmente "id". M/B junto com medidas
    # numéricas indica um registro; não descartar essa primeira amostra.
    if any(value.strip() in {"M", "B"} for value in first) and any(
            _is_number(value, delimiter) for value in first):
        return False
    if any(_name_key(value) in _ID_NAMES | _TARGET_NAMES for value in first):
        return True
    if len(rows) < 2 or any(value.strip() in {"M", "B"} for value in first):
        return False
    for index, value in enumerate(first):
        rest = [row[index].strip() for row in rows[1:] if row[index].strip()]
        if value.strip() and not _is_number(value, delimiter) and rest:
            if all(_is_number(cell, delimiter) for cell in rest):
                return True
            if set(rest) <= {"M", "B"}:
                return True
    return False


def inspect_source(text: str, delimiter: str = "auto", header: str = "auto",
                   feature_names: list[str] | None = None) -> dict:
    """Retorna todas as células e uma prévia de até oito valores por coluna.

    O reconhecimento de cabeçalho é uma sugestão ajustável. Sem cabeçalho,
    metadados numéricos genéricos precisam ser identificados pelo usuário.
    """
    if not isinstance(text, str) or not text.strip().lstrip("\ufeff"):
        raise ValueError("Arquivo vazio: nenhum registro encontrado.")
    if not isinstance(header, str) or header not in {"auto", "yes", "no"}:
        raise ValueError("Cabeçalho deve ser auto, yes ou no.")
    if not isinstance(delimiter, str) or (delimiter != "auto" and delimiter not in _DELIMITERS):
        raise ValueError("Separador deve ser auto, vírgula, ponto e vírgula, TAB ou whitespace (espaços).")
    payload, offset, wdbc = _wdbc_payload(text.lstrip("\ufeff"))
    protein = False
    if not wdbc:
        payload, offset, protein = _protein_payload(payload)
    profile_name = None
    if not wdbc and not protein:
        payload, offset, profile_name = _tabular_profile_payload(payload)
    profile = _TABULAR_PROFILES.get(profile_name)
    separator = ("whitespace" if protein else "," if profile else _detect_delimiter(payload)) if delimiter == "auto" else _DELIMITERS[delimiter]
    if wdbc and (separator != "," or header == "yes"):
        raise ValueError("O TXT WDBC conhecido usa vírgula, sem cabeçalho nos registros.")
    if protein and (separator != "whitespace" or header == "yes"):
        raise ValueError("O TXT Protein Localization Sites conhecido usa espaços (whitespace), sem cabeçalho nos registros.")
    if profile and (separator != "," or header == "yes"):
        raise ValueError(f"O TXT {profile['label']} conhecido usa vírgula, sem cabeçalho nos registros.")
    rows, source_lines = _records(payload, separator, offset)
    if not rows:
        raise ValueError("Nenhum registro encontrado no arquivo.")
    width = len(rows[0])
    if not 1 <= width <= 512:
        raise ValueError("O arquivo deve conter de 1 a 512 colunas.")
    if protein and width != 10:
        raise ValueError(f"Linha {source_lines[0]}: TXT Protein Localization Sites exige 10 colunas "
                         f"(ID, 8 medidas e classe); encontradas {width}.")
    if profile and width != len(profile["names"]):
        raise ValueError(f"Linha {source_lines[0]}: TXT {profile['label']} exige {len(profile['names'])} colunas; encontradas {width}.")
    for row, line in zip(rows, source_lines):
        if len(row) != width:
            raise ValueError(f"Linha {line}: esperadas {width} colunas; encontradas {len(row)}.")
    if wdbc and width != 32:
        raise ValueError(f"TXT WDBC original exige 32 colunas; encontradas {width}.")
    has_header = False if wdbc or protein or profile else (header == "yes" or (header == "auto" and _has_header(rows, separator)))
    original_names = [value.strip() for value in rows[0]] if has_header else []
    if has_header:
        rows, source_lines = rows[1:], source_lines[1:]
    if not rows:
        raise ValueError("O arquivo contém cabeçalho, mas nenhum registro de dados.")
    warnings = []
    if wdbc:
        names = list(feature_names) if feature_names is not None else WDBC_FEATURES.copy()
        if len(names) != 30 or any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != 30:
            raise ValueError("O WDBC requer exatamente 30 nomes de características únicos.")
        original_names = ["id", "diagnosis", *names]
    elif protein:
        original_names = ["id", *PROTEIN_FEATURES, "classe"]
    elif profile:
        original_names = profile["names"].copy()
        if profile_name == "abalone_gender":
            warnings.append("Nesta base, o alvo escolhido é M/F/I em x1: I significa infantil. "
                            "Rings permanece entrada; a documentação original descreve outro objetivo, prever anéis/idade.")
    if not original_names:
        original_names = [f"x{i + 1}" for i in range(width)]
    counts = Counter(original_names)
    names = [name if name and counts[name] == 1 else f"x{i + 1}"
             for i, name in enumerate(original_names)]
    # Colisões com xN explícito têm uma alternativa estável por posição.
    while len(set(names)) != width:
        collisions = {name for name, count in Counter(names).items() if count > 1}
        names = [f"x{i + 1}" if name in collisions else name for i, name in enumerate(names)]
    if names != original_names:
        warnings.append("Cabeçalhos vazios, repetidos ou em conflito foram identificados pelo nome estável xN.")
    columns, label_candidates = [], []
    for index, name in enumerate(names):
        values = [row[index] for row in rows]
        present = [value.strip() for value in values if value.strip()]
        numeric = bool(present) and all(_is_number(value, separator) for value in present)
        # Use o cabeçalho original para reconhecer metadados mesmo se repetido.
        key = _name_key(original_names[index])
        role = "id" if key in _ID_NAMES else "target" if key in _TARGET_NAMES else "feature" if numeric else "ignore"
        if profile:
            # Layout documentado: uma medida inválida não vira silenciosamente
            # coluna ignorada. A seleção exigirá corrigi-la ou ignorá-la.
            role = "target" if index == profile["target"] else "feature"
        if role not in {"id", "target"} and present and set(present) <= {"M", "B"}:
            label_candidates.append(index)
        columns.append({"index": index, "key": f"x{index + 1}", "name": name,
                        "description": (_PROTEIN_DESCRIPTIONS[index] if protein else profile["descriptions"][index]
                                        if profile else original_names[index] or f"x{index + 1}"),
                        "role": role, "numeric": numeric,
                        "missing": len(values) - len(present), "preview": values[:8]})
    named_targets = [column for column in columns if column["role"] == "target"]
    if not named_targets and len(label_candidates) == 1:
        columns[label_candidates[0]]["role"] = "target"
    targets = [column for column in columns if column["role"] == "target"]
    target_summary = None
    if len(targets) != 1:
        warnings.append("Não há uma única coluna de alvo inequívoca; escolha a coluna com as classes.")
    else:
        target = targets[0]
        classes = Counter(row[target["index"]].strip() for row in rows)
        supported = len(classes) >= 2 and all(label and len(label) <= 100 for label in classes)
        target_summary = {"index": target["index"], "key": target["key"],
                          "classes": [{"label": label, "count": count} for label, count in classes.items()],
                          "supported": supported}
        if not supported:
            warnings.append(f"Alvo {target['key']}: selecione pelo menos duas classes não vazias, "
                            "com rótulos de até 100 caracteres. A validade do protocolo ainda precisa ser conferida.")
    if sum(column["role"] == "id" for column in columns) > 1:
        warnings.append("Mais de uma coluna de ID reconhecida; escolha no máximo uma.")
    ignored = [column["key"] for column in columns if column["role"] == "ignore"]
    if ignored:
        warnings.append("Colunas vazias, textuais ou com valores não finitos foram sugeridas como ignoradas: " + ", ".join(ignored) + ".")
    return {"delimiter": separator, "has_header": has_header,
            "source_format": "protein_localization" if protein else "wdbc" if wdbc else profile_name or "delimited",
            "target_summary": target_summary,
            "source_lines": source_lines, "rows": rows, "columns": columns,
            "n_rows": len(rows), "n_columns": width, "warnings": warnings}


def selected_data(table: dict, schema: dict) -> dict:
    """Valida a seleção recebida e extrai X e y sem alterações no arquivo."""
    if not isinstance(schema, dict):
        raise ValueError("O esquema de colunas deve ser um objeto.")
    separator = schema.get("delimiter")
    if not isinstance(separator, str) or separator not in _DELIMITERS or _DELIMITERS[separator] != table["delimiter"]:
        raise ValueError("Separador do esquema difere do arquivo inspecionado; inspecione novamente.")
    expected_header = "yes" if table["has_header"] else "no"
    if schema.get("header") != expected_header:
        raise ValueError("Cabeçalho do esquema difere do arquivo inspecionado; inspecione novamente.")
    supplied = schema.get("columns")
    width = table["n_columns"]
    if not isinstance(supplied, list) or len(supplied) != width:
        raise ValueError("O esquema deve identificar todas as colunas exatamente uma vez.")
    by_index = {}
    for item in supplied:
        if not isinstance(item, dict):
            raise ValueError("Cada coluna do esquema deve ser um objeto.")
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < width or index in by_index:
            raise ValueError("Índices das colunas inválidos ou repetidos.")
        role, description = item.get("role"), item.get("description")
        if not isinstance(role, str) or role not in _ROLES:
            raise ValueError(f"Papel inválido para x{index + 1}.")
        if not isinstance(description, str) or len(description) > 1000:
            raise ValueError(f"Descrição de x{index + 1} deve ser texto com até 1000 caracteres.")
        original = table["columns"][index]
        if role == "feature" and (original["role"] in {"id", "target"} or
                                  _name_key(original["name"]) in _ID_NAMES | _TARGET_NAMES):
            raise ValueError(f"{original['key']}: ID/diagnóstico reconhecido não pode ser entrada da rede.")
        by_index[index] = {"index": index, "description": description, "role": role}
    normalized_columns = [by_index[index] for index in range(width)]
    target_indices = [column["index"] for column in normalized_columns if column["role"] == "target"]
    id_indices = [column["index"] for column in normalized_columns if column["role"] == "id"]
    feature_indices = [column["index"] for column in normalized_columns if column["role"] == "feature"]
    if len(target_indices) != 1:
        raise ValueError("Selecione exatamente uma coluna de diagnóstico/alvo.")
    if len(id_indices) > 1:
        raise ValueError("Selecione no máximo uma coluna de ID.")
    if not 1 <= len(feature_indices) <= 256:
        raise ValueError("Selecione de 1 a 256 características de entrada.")
    features = [table["columns"][index]["name"] for index in feature_indices]
    if len(set(features)) != len(features) or any(not name or len(name) > 100 for name in features):
        raise ValueError("Características requerem nomes únicos com 1 a 100 caracteres.")
    descriptions = [by_index[index]["description"] for index in feature_indices]
    labels = Counter(row[target_indices[0]].strip() for row in table["rows"])
    if any(not label or len(label) > 100 for label in labels):
        invalid_index = next(index for index, row in enumerate(table["rows"])
                             if not row[target_indices[0]].strip() or len(row[target_indices[0]].strip()) > 100)
        line = table["source_lines"][invalid_index]
        label = table["rows"][invalid_index][target_indices[0]].strip()
        raise ValueError(f"Linha {line}: diagnóstico/alvo deve ter rótulo não vazio com até 100 caracteres; encontrado {label!r}.")
    classes = ["M", "B"] if set(labels) <= {"M", "B"} else sorted(labels)
    if len(classes) < 2:
        raise ValueError("O alvo requer pelo menos duas classes.")
    class_index = {label: index for index, label in enumerate(classes)}
    legacy_binary = classes == ["M", "B"]
    X, y, ids, seen_ids = [], [], [], set()
    for row, line in zip(table["rows"], table["source_lines"]):
        identifier = row[id_indices[0]].strip() if id_indices else f"linha_{line}"
        if not identifier or (legacy_binary and identifier in seen_ids):
            raise ValueError(f"Linha {line}: ID vazio ou repetido; revise a coluna de identificação.")
        seen_ids.add(identifier)
        label = row[target_indices[0]].strip()
        values = []
        for index in feature_indices:
            try:
                values.append(_number(row[index], table["delimiter"]))
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Linha {line}, x{index + 1}: medida vazia, não numérica ou não finita.") from exc
        X.append(values)
        y.append(class_index[label])
        ids.append(identifier)
    return {"X": X, "y": y, "ids": ids, "features": features, "classes": classes,
            "descriptions": descriptions, "source_lines": table["source_lines"].copy(),
            "schema": {"delimiter": table["delimiter"], "header": expected_header,
                       "columns": normalized_columns}}
