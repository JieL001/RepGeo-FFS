$FfsPython = "F:\miniforge3\envs\ffs\python.exe"

if (Test-Path $FfsPython) {
  $script:DemoPython = $FfsPython
} else {
  $script:DemoPython = "python"
}

Write-Host "Python interpreter: $script:DemoPython"
