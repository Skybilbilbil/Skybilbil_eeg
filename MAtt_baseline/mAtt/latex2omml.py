#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LaTeX 子集 -> OMML（Word 原生公式）转换器。

覆盖讲义里实际用到的所有构造：
  分式 \frac、根式 \sqrt、上下标 ^ _、定界符 \left..\right、
  矩阵 \begin{bmatrix}..\end{bmatrix}、文本 \text \mathrm \mathbb \mathbf、
  函数名 \log \ln \exp 等、求和 \sum、希腊字母、常用运算符与空格命令。

对外只暴露 latex_to_omml(latex) -> XML 字符串（含 <m:oMath> 外壳）。
解析失败时抛异常，由调用方回退。
"""
import re

M_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

GREEK = {
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε',
    'varepsilon': 'ε', 'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'iota': 'ι',
    'kappa': 'κ', 'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'pi': 'π',
    'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ', 'phi': 'φ',
    'varphi': 'φ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
    'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ', 'Xi': 'Ξ',
    'Pi': 'Π', 'Sigma': 'Σ', 'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω',
}

OPS = {
    'cdot': '·', 'times': '×', 'div': '÷', 'pm': '±', 'mp': '∓',
    'le': '≤', 'leq': '≤', 'ge': '≥', 'geq': '≥', 'ne': '≠', 'neq': '≠',
    'approx': '≈', 'equiv': '≡', 'sim': '∼', 'propto': '∝',
    'to': '→', 'rightarrow': '→', 'Rightarrow': '⇒', 'leftarrow': '←',
    'Leftarrow': '⇐', 'leftrightarrow': '↔', 'mapsto': '↦',
    'in': '∈', 'notin': '∉', 'subset': '⊂', 'subseteq': '⊆', 'cup': '∪',
    'cap': '∩', 'forall': '∀', 'exists': '∃', 'infty': '∞', 'partial': '∂',
    'nabla': '∇', 'top': 'ᵀ', 'perp': '⊥', 'parallel': '∥', 'circ': '∘',
    'star': '⋆', 'ast': '∗', 'ldots': '…', 'cdots': '⋯', 'dots': '…',
    'langle': '⟨', 'rangle': '⟩', 'lvert': '|', 'rvert': '|',
    'lVert': '‖', 'rVert': '‖', 'vert': '|', 'Vert': '‖',
    'quad': '\u2003', 'qquad': '\u2003\u2003', 'colon': '：',
    'checkmark': '✓', 'prime': '′', 'triangleq': '≜', 'iff': '⟺',
}

BLACKBOARD = {'R': 'ℝ', 'N': 'ℕ', 'Z': 'ℤ', 'Q': 'ℚ', 'C': 'ℂ', 'P': 'ℙ'}

FUNCS = {'log', 'ln', 'exp', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
         'max', 'min', 'arg', 'det', 'tr', 'dim', 'rank', 'sup', 'inf',
         'lim', 'gcd', 'Pr'}

SPACES = {' ', '\u2003', ',', ';', ':', '!'}

DELIM = {'(': '(', ')': ')', '[': '[', ']': ']', r'\{': '{', r'\}': '}',
         '|': '|', r'\|': '‖', '.': '', '<': '⟨', '>': '⟩'}


def _run(text, upright=False, bold=False):
    rpr = ''
    if upright or bold:
        rpr = '<m:rPr>'
        if bold:
            rpr += '<m:sty m:val="b"/>'
        elif upright:
            rpr += '<m:sty m:val="p"/>'
        rpr += '</m:rPr>'
    return ('<m:r>' + rpr + '<m:t xml:space="preserve">' +
            _esc(text) + '</m:t></m:r>')


def _esc(t):
    return (t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def _frac(num, den):
    return ('<m:f><m:fPr><m:type m:val="bar"/></m:fPr>'
            '<m:num>' + num + '</m:num>'
            '<m:den>' + den + '</m:den></m:f>')


def _rad(e, deg=''):
    return ('<m:rad><m:radPr><m:degHide m:val="' +
            ('1' if not deg else '0') + '"/></m:radPr>'
            '<m:deg>' + deg + '</m:deg><m:e>' + e + '</m:e></m:rad>')


def _delim(inner, beg='(', end=')'):
    return ('<m:d><m:dPr><m:begChr m:val="' + _esc(beg) + '"/>'
            '<m:endChr m:val="' + _esc(end) + '"/></m:dPr>'
            '<m:e>' + inner + '</m:e></m:d>')


def _matrix(rows, ncol):
    mc = ('<m:mc><m:mcPr><m:count m:val="' + str(ncol) + '"/>'
          '<m:mcJc m:val="center"/></m:mcPr></m:mc>')
    body = ''
    for row in rows:
        cells = ''.join('<m:e>' + c + '</m:e>' for c in row)
        body += '<m:mr>' + cells + '</m:mr>'
    return ('<m:m><m:mPr><m:mcs>' + mc + '</m:mcs></m:mPr>' + body + '</m:m>')


def _nary(chr_, sub, sup, e):
    return ('<m:nary><m:naryPr><m:chr m:val="' + chr_ + '"/>'
            '<m:limLoc m:val="undOvr"/><m:subHide m:val="' +
            ('1' if not sub else '0') + '"/><m:supHide m:val="' +
            ('1' if not sup else '0') + '"/></m:naryPr>'
            '<m:sub>' + sub + '</m:sub><m:sup>' + sup + '</m:sup>'
            '<m:e>' + e + '</m:e></m:nary>')


class _P:
    def __init__(self, s):
        self.s = s
        self.i = 0
        self.steps = 0

    def eof(self):
        return self.i >= len(self.s)

    def peek(self, k=0):
        j = self.i + k
        return self.s[j] if j < len(self.s) else ''

    def skip_ws(self):
        while not self.eof() and self.s[self.i] in ' \t\n':
            self.i += 1

    def read_cmd(self):
        # 当前位置是反斜杠
        self.i += 1
        m = re.match(r'[A-Za-z]+', self.s[self.i:])
        if m:
            self.i += len(m.group(0))
            return m.group(0)
        if self.eof():
            return ''
        ch = self.s[self.i]
        self.i += 1
        return ch      # \{ \} \| \, \; 等

    def read_group(self):
        """读一个参数：{...} 或单个原子。返回 OMML 片段。"""
        self.skip_ws()
        if self.peek() == '{':
            j = self._match_brace(self.i)
            inner = self.s[self.i + 1:j]
            self.i = j + 1
            return self.parse(inner)
        return self.parse_primitive()

    def _match_brace(self, start):
        depth = 0
        j = start
        while j < len(self.s):
            if self.s[j] == '{' and (j == 0 or self.s[j - 1] != '\\'):
                depth += 1
            elif self.s[j] == '}' and (j == 0 or self.s[j - 1] != '\\'):
                depth -= 1
                if depth == 0:
                    return j
            j += 1
        return len(self.s) - 1

    def parse_primitive(self):
        """解析一个不可再分的基元，返回 OMML。"""
        self.skip_ws()
        if self.eof():
            return ''
        c = self.peek()

        if c == '{':
            j = self._match_brace(self.i)
            inner = self.s[self.i + 1:j]
            self.i = j + 1
            return self.parse(inner)

        if c == '\\':
            save = self.i
            cmd = self.read_cmd()
            return self._cmd_to_omml(cmd, save)

        # 普通字符
        self.i += 1
        if c in '{}':
            return ''
        if c == '~':
            return _run('\u00a0')
        return _run(c)

    def _cmd_to_omml(self, cmd, save):
        # 分式
        if cmd in ('frac', 'dfrac', 'tfrac'):
            num = self.read_group()
            den = self.read_group()
            return _frac(num, den)

        # 根式
        if cmd == 'sqrt':
            deg = ''
            self.skip_ws()
            if self.peek() == '[':
                j = self.s.find(']', self.i)
                deg = self.parse(self.s[self.i + 1:j])
                self.i = j + 1
            body = self.read_group()
            return _rad(body, deg)

        # 下括注 underbrace / 上括注 overbrace
        if cmd in ('underbrace', 'overbrace'):
            e = self.read_group()
            lim = ''
            self.skip_ws()
            if self.peek() == '_' or self.peek() == '^':
                self.i += 1
                lim = self.read_group()
            if not lim:
                return e
            tag = 'limLow' if cmd == 'underbrace' else 'limUpp'
            return ('<m:' + tag + '><m:e>' + e + '</m:e><m:lim>' +
                    lim + '</m:lim></m:' + tag + '>')

        # 文本类
        if cmd in ('text', 'mathrm', 'mathit', 'mathbf', 'mathcal',
                   'mathsf', 'operatorname', 'mathbb'):
            self.skip_ws()
            if self.peek() != '{':
                return _run('')
            j = self._match_brace(self.i)
            raw = self.s[self.i + 1:j]
            self.i = j + 1
            if cmd == 'mathbb':
                conv = ''.join(BLACKBOARD.get(ch, ch) for ch in raw)
                return _run(conv)
            if cmd in ('mathbf',):
                return _run(raw, bold=True)
            if cmd in ('mathit', 'mathcal'):
                return _run(raw)
            return _run(raw, upright=True)

        # 定界符
        if cmd == 'left':
            self.skip_ws()
            beg = ''
            if self.peek() == '\\':
                b = self.read_cmd()
                beg = DELIM.get('\\' + b, b)
            else:
                ch = self.peek()
                self.i += 1
                beg = DELIM.get(ch, ch)
            inner = self.parse_until_right()
            self.skip_ws()
            end = ''
            if not self.eof() and self.peek() == '\\':
                save2 = self.i
                c2 = self.read_cmd()
                if c2 == 'right':
                    self.skip_ws()
                    if self.peek() == '\\':
                        e = self.read_cmd()
                        end = DELIM.get('\\' + e, e)
                    else:
                        ch = self.peek()
                        self.i += 1
                        end = DELIM.get(ch, ch)
                else:
                    self.i = save2
            return _delim(inner, beg, end)

        if cmd == 'right':
            return ''

        # 矩阵等环境
        if cmd in ('begin', 'end'):
            self.skip_ws()
            if self.peek() == '{':
                j = self._match_brace(self.i)
                env = self.s[self.i + 1:j]
                self.i = j + 1
            else:
                env = ''
            if cmd == 'end':
                return ''
            if env in ('bmatrix', 'pmatrix', 'matrix', 'vmatrix'):
                rows_raw = self._read_until_env_end(env)
                rows = []
                for line in re.split(r'\\\\', rows_raw):
                    line = line.strip()
                    if not line:
                        continue
                    rows.append([self.parse(x) for x in line.split('&')])
                if not rows:
                    return ''
                ncol = max(len(r) for r in rows)
                rows = [r + [''] * (ncol - len(r)) for r in rows]
                m = _matrix(rows, ncol)
                if env == 'bmatrix':
                    return _delim(m, '[', ']')
                if env == 'pmatrix':
                    return _delim(m, '(', ')')
                if env == 'vmatrix':
                    return _delim(m, '|', '|')
                return m
            return ''

        # 求和 / 连乘
        if cmd in ('sum', 'prod'):
            sub, sup = self._read_limits()
            body = self.parse_primitive()
            return _nary('∑' if cmd == 'sum' else '∏', sub, sup, body)

        # 函数名
        if cmd in FUNCS:
            return _run(cmd, upright=True)

        # 希腊字母
        if cmd in GREEK:
            return _run(GREEK[cmd])

        # 运算符 / 空格 / 其他
        if cmd in OPS:
            v = OPS[cmd]
            return _run(v)

        if cmd in (' ', ',', ';', ':', '!'):
            return _run('\u2009')

        if cmd in ('{', '}', '|'):
            return _run(cmd)

        if cmd == '\\':
            return _run('\u2003')

        # 未识别：退化为字母
        return _run(cmd)

    def _read_limits(self):
        sub = sup = ''
        for _ in range(2):
            self.skip_ws()
            if self.peek() == '_':
                self.i += 1
                sub = self.read_group()
            elif self.peek() == '^':
                self.i += 1
                sup = self.read_group()
            else:
                break
        return sub, sup

    def _read_until_env_end(self, env):
        tag = r'\end{' + env + '}'
        j = self.s.find(tag, self.i)
        if j < 0:
            j = len(self.s)
            out = self.s[self.i:]
            self.i = len(self.s)
            return out
        out = self.s[self.i:j]
        self.i = j + len(tag)
        return out

    def parse_until_right(self):
        """读到 \\right 之前（不消费 \\right）。"""
        j = self.s.find(r'\right', self.i)
        if j < 0:
            out = self.s[self.i:]
            self.i = len(self.s)
            return self.parse(out)
        out = self.s[self.i:j]
        self.i = j
        return self.parse(out)

    def parse(self, s=None):
        """解析一段字符串为 OMML 序列（含上下标装配）。"""
        if s is not None:
            sub = _P(s)
            return sub.parse()
        items = []
        while True:
            self.steps += 1
            if self.steps > 20000:
                raise RuntimeError('parse loop runaway at %r' % self.s[self.i:self.i + 40])
            self.skip_ws()
            if self.eof():
                break
            if self.peek() == '}':
                break
            base = self.parse_primitive()
            sup = sub_ = None
            while True:
                self.skip_ws()
                if self.peek() == '^':
                    self.i += 1
                    sup = self.read_group()
                elif self.peek() == '_':
                    self.i += 1
                    sub_ = self.read_group()
                else:
                    break
            items.append((base, sub_, sup))
        out = []
        for base, sb, sp in items:
            if sb and sp:
                out.append('<m:sSubSup><m:e>' + base + '</m:e><m:sub>' + sb +
                           '</m:sub><m:sup>' + sp + '</m:sup></m:sSubSup>')
            elif sp:
                out.append('<m:sSup><m:e>' + base + '</m:e><m:sup>' + sp +
                           '</m:sup></m:sSup>')
            elif sb:
                out.append('<m:sSub><m:e>' + base + '</m:e><m:sub>' + sb +
                           '</m:sub></m:sSub>')
            else:
                out.append(base)
        return ''.join(out)


def latex_to_omml(latex):
    body = _P(latex).parse()
    return ('<m:oMath xmlns:m="' + M_NS + '" xmlns:w="' + W_NS + '">' +
            body + '</m:oMath>')


if __name__ == '__main__':
    tests = [
        r'\frac{n(n+1)}{2}',
        r'\sqrt{34} \approx 5.83',
        r'A = \begin{bmatrix} a & b \\ b & d \end{bmatrix}',
        r'v^{\top} C v \ge 0',
        r'\log\left(A^{-1/2} B A^{-1/2}\right)',
        r'\sum_{t=1}^{T}\bigl(x_t - \mu\bigr)^{2}',
        r'\mathbb{R}^{n(n+1)/2}',
        r'\mathrm{tr}(C^{-1} U C^{-1} V)',
    ]
    for t in tests:
        try:
            x = latex_to_omml(t)
            ok = 'f' if '<m:f>' in x else ('rad' if '<m:rad>' in x else '-')
            ok2 = 'm' if '<m:m>' in x else '-'
            ok3 = 'd' if '<m:d>' in x else '-'
            print('OK %-52s %s %s %s' % (t[:50], ok, ok2, ok3))
        except Exception as e:
            print('FAIL', t[:50], '->', e)
