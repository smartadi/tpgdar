# Turn a folder of numbered PNG frames, in a helper .pptx with auto-advance, into an MP4 through
# desktop PowerPoint.  No ffmpeg on this PC, and PowerPoint for the web plays video but not GIFs.
param([string]$deck, [string]$out, [int]$height = 720, [int]$fps = 30)
$p = New-Object -ComObject PowerPoint.Application
$d = $p.Presentations.Open($deck, -1, 0, 0)
$d.CreateVideo($out, -1, 1, $height, $fps, 85)
# 1 queued, 2 in progress, 3 done, 4 failed
$tries = 0
while ($d.CreateVideoStatus -lt 3 -and $tries -lt 900) {
    Start-Sleep -Seconds 2
    $tries++
}
$status = $d.CreateVideoStatus
$d.Close()
$p.Quit()
"status $status after $($tries * 2) s"
