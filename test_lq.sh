#!/bin/sh

test_config="test_config_$$.json"
test_image="test_image.$$.gif"
umask 066
cat > $test_config <<_EOF_
{
  "defaults": {
    "model_name":"gemma4"
  },
  "models": [
    {
      "name":"gemma4",
      "api_url": "http://127.0.0.1:11434/v1",
      "model": "gemma-4-e4b-it"
    }
  ],
  "templates": [
    {
      "name": "t0",
      "prompt": "say hello"
    },
    {
      "name": "t1",
      "prompt": "say %s",
      "defaults": ["hello"]
    },
    {
      "name": "t2",
      "prompt": "say %s %s times",
      "defaults": [null, 3]
    }
  ]
}
_EOF_
base64 -d <<_EOF_ > $test_image
R0lGODlhFAAUAPABAAAAAP///yH5BAAAAAAAIf8LSW1hZ2VNYWdpY2sOZ2FtbWE9MC40NT
Q1NDUALAAAAAAUABQAAAIyjI+pCw0Lg3tRzZpmw2dLynke5hglCX5hekLtl1qxGmm2a2v4
zsym/0IEaTyhrzPiLAoAOw==
_EOF_

trap "rm -f $test_config $test_image" EXIT

lq="python3 lq.py -c $test_config"

# do test: T "test name" expected_result "test command"
T(){
  printf "$1: "
  expected=$2
  shift 2
  eval "$@" >/dev/null 2>/dev/null
  case $? in
  $expected) success=$((success+1)); echo ok;;
  *)         fail=$((fail+1)); echo fail;;
  esac
}

success=0
fail=0

T "simple query" 0 '$lq "say hello" | grep -i hello'
T "file attachment" 0 '$lq -f $test_config "what type is this file?" | grep -i json'
T "image attachment" 0 '$lq -i $test_image "what letter?" | grep -o A'
T "read from stdin" 0 '$lq "what type is this file?" < $test_config | grep -i json'
T "no prompt" 1 '$lq -f $ test_config'
T "json output" 0 '$lq --json "say hello" | jq -e .object==\"chat.completion\"'
T "template without parameter" 0 '$lq -t t0 | grep -i hello'
T "template with default parameter" 0 '$lq -t t1 | grep -i hello'
T "template with parameter from arg" 0 '$lq -t t1 "good morning" | grep -i "good morning"'
T "template with 2 parameters" 0 't=$($lq -t t2 hello 3 | xargs -n1 | grep -ioc hello); [ $t -eq 3 ]'
T "template with too many params" 1 '$lq -t t1 foo bar'
T "template with too few params" 1 '$lq -t t2'
T "chat without tty" 1 '$lq --chat < $test_config'
T "ignore chat" 0 '$lq --chat "what type is this file?" < $test_config 2>&1 | grep Warning'

chmod 644 $test_config
T "config permission" 0 '$lq say hello 2>&1 | grep Warning'

echo
echo Success: $success, Fail: $fail
