param(
  [Parameter(Mandatory = $true)]
  [string]$Path,
  [int]$Head = 0,
  [int]$Tail = 0
)

# Ensure UTF-8 output without BOM so the harness renders Unicode correctly
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$ErrorActionPreference = 'Stop'

if ($Head -gt 0 -and $Tail -gt 0) {
  throw "Specify only one of -Head or -Tail"
}

if ($Head -gt 0) {
  Get-Content -LiteralPath $Path -Encoding UTF8 -TotalCount $Head
} elseif ($Tail -gt 0) {
  Get-Content -LiteralPath $Path -Encoding UTF8 -Tail $Tail
} else {
  Get-Content -LiteralPath $Path -Encoding UTF8
}

