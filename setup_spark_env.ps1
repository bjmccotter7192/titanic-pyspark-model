# Set environment variables for the current session
$env:HADOOP_HOME = "C:\hadoop"
$env:PATH = "$env:PATH;$env:HADOOP_HOME\bin"
$env:JAVA_HOME = $env:JAVA_HOME

Write-Host "Environment variables set:"
Write-Host "HADOOP_HOME: $env:HADOOP_HOME"
Write-Host "JAVA_HOME: $env:JAVA_HOME"
Write-Host "PATH updated to include Hadoop bin directory"
