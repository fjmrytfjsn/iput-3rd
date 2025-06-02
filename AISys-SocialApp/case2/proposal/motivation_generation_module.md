```mermaid
flowchart TD
    A[学生入力情報]
    B[AI生成API]
    C[企業特徴データ]
    D[志望動機テンプレート]
    E[志望動機案（生成結果）]

    A --> B
    C --> B
    D --> B
    B --> E
```
