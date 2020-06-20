package com.mycompany.myproject.utils;

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import lombok.extern.slf4j.Slf4j;
import net.sourceforge.pinyin4j.PinyinHelper;
import net.sourceforge.pinyin4j.format.HanyuPinyinCaseType;
import net.sourceforge.pinyin4j.format.HanyuPinyinOutputFormat;
import net.sourceforge.pinyin4j.format.HanyuPinyinToneType;
import net.sourceforge.pinyin4j.format.exception.BadHanyuPinyinOutputFormatCombination;
import org.apache.commons.lang3.StringUtils;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

@Service
@Slf4j
public class PinyinUtils {
    private static HanyuPinyinOutputFormat pinyinFormater;
    static {
        pinyinFormater = new HanyuPinyinOutputFormat();
        pinyinFormater.setCaseType(HanyuPinyinCaseType.LOWERCASE);
        pinyinFormater.setToneType(HanyuPinyinToneType.WITHOUT_TONE);
    }
    public static String mapHanziToPinyin(String hanzi) {
        if (StringUtils.isBlank(hanzi)) {
            return null;
        }
        List<Character> chars = hanzi.codePoints().mapToObj(c -> (char) c).collect(Collectors.toList());
        List<String> pinyinList = chars.stream().map(PinyinUtils::toPinyin).collect(Collectors.toList());
        return Joiner.on("").join(pinyinList);
    }
    private static String toPinyin(Character hanzi) {
        String py;
        try {
            String[] pys = PinyinHelper.toHanyuPinyinStringArray(hanzi, pinyinFormater);
            if (Objects.isNull(pys)) {
                return hanzi.toString();
            }
            py = Lists.newArrayList(pys).stream().findFirst().orElse("");
        } catch (BadHanyuPinyinOutputFormatCombination badHanyuPinyinOutputFormatCombination) {
            throw new RuntimeException("toPinyin failed");
        }
        return py;
    }
}