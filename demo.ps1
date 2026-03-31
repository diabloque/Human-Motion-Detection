param(
    [ValidateSet("train", "predict", "webcam")]
    [string]$Mode = "predict",

    [string]$Data = "datasets/helmet_data.yaml",
    [string]$Model = "runs/helmet_50ep/weights/best.pt",
    [string]$Source = "results/sample_predictions",
    [int]$Epochs = 50,
    [int]$ImgSize = 640,
    [int]$Batch = 8,
    [double]$Conf = 0.25,
    [string]$RunName = "demo_run"
)

$ErrorActionPreference = "Stop"

function Get-Python {
    if (Test-Path ".\.venv\Scripts\python.exe") {
        return ".\.venv\Scripts\python.exe"
    }
    return "python"
}

$python = Get-Python

Write-Host "Mode: $Mode"
Write-Host "Python: $python"

switch ($Mode) {
    "train" {
        & $python "train.py" `
            --data "$Data" `
            --epochs $Epochs `
            --imgsz $ImgSize `
            --batch $Batch `
            --name $RunName
    }
    "predict" {
        & $python "predict.py" `
            --model "$Model" `
            --source "$Source" `
            --imgsz $ImgSize `
            --conf $Conf `
            --name $RunName
    }
    "webcam" {
        & $python "predict.py" `
            --model "$Model" `
            --source 0 `
            --imgsz $ImgSize `
            --conf $Conf `
            --name $RunName `
            --show
    }
}

