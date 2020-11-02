import json

def main():
    data_file = "online_test_data/test_answers.json"
    with open(data_file) as f:
        data = json.load(f)
    ans_dict = {}
    for ans in data:
        text = ans["Answer"]
        if text not in ans_dict:
            ans_dict[text] = 1
        else:
            ans_dict[text] += 1
    ans_list = list(ans_dict.items())
    ans_list = sorted(ans_list, key=lambda x: x[1], reverse=True)
    with open("ans_c.txt", "w") as f:
        for text, count in ans_list:
            f.write(F"{count} {text}\n")
    pass

if __name__ == "__main__":
    main()
