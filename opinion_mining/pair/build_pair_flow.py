from pair import build_candidate
from pair import pair_refine
from pair import pair_patt_sort
from pair import pair_mine
from pprint import pprint

def build_pair(domain, mongodb_client, resources, aspect_for_filter):

    collection = mongodb_client.db['opinion_build_pairs']

    """《大规模中文实体情感知识的自动获取》卢奇，陈文亮"""
    """1.抽取候选pair"""
    buildset = build_candidate.AspectFilter(resources.seg_pu_list, resources.pos_pu_list)
    #ns_dict每次不一样导致的结果不一样，ns_dict每次的一样导致结果一样
    ns_dict = buildset.build_nsdict(aspect_for_filter)

    """2.组合排序"""
    setsort = pair_patt_sort.PairPattSort(ns_dict)
    pair_sorted_ns = setsort.pair_sort()  # <- 得到排序过后的分数

    """3.pair提炼"""
    pairmine = pair_mine.PairMine(pair_sorted_ns, resources.word2vec_model)
    pair_mine_aspect = pairmine.aspect_mine()
    pair_mine_opinion = pairmine.opinion_mine()

    """4.得到最终结果"""
    pairrefine = pair_refine.PairRefine(resources.general_opinion, pair_sorted_ns, pair_mine_aspect, pair_mine_opinion)
    refine_pair = pairrefine.refine()
    pair_polarity_list = pairrefine.gen_pair_polarity(refine_pair)
    different_dict, size = pairrefine.gen_final_build_result_and_save_in_db(domain, collection, pair_polarity_list)
    m_res_str = pairrefine.merge_with_annotation(mongodb_client, domain)
    return 'build结束共得到{}个表达,与人工标注的极性pair进行合并结果:{}'.format(size, m_res_str), different_dict

