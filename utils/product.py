import jieba

class ProductInfo(object):
    def __init__(self, kb_f):
        self.kb_f = kb_f
        self.product_infos = {}
        self.load()
        pass

    def load(self):
        import json
        with open(self.kb_f) as f:
            infos = json.load(f)
        for index, info in enumerate(infos):
            # content = list(str(index))
            content = []
            pid = info["pid"]
            # for k, v in info.items():
            #     if k == "pid":
            #         continue
            #     content.extend(jieba.cut(k, cut_all=False))
            #     content.extend(jieba.cut(v, cut_all=False))
            value = info["分类"]
            content = value  # list(jieba.cut(value, cut_all=False))
            self.product_infos[pid] = content

    def get_info_by_pid(self, pid=None):
        if not pid:
            return "未知"
        if pid not in self.product_infos:
            return "未知"
        # return pid + " " + " ".join(self.product_infos[pid]) + " <$$$> "
        return self.product_infos[pid]
        # return " ".join(product_infos[pid])
